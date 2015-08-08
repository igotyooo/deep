classdef NeuralDscrber < handle
    properties
        db;
        net;
        setting;
    end
    methods
        function this = NeuralDscrber( db, net, setting )
            this.db = db;
            this.net = net;
            this.setting.layerId = numel( net.layers ) - 2;     % FC7 in AlexNet.
            this.setting.augmentationType = 'NONE';             % F5, F25, NONE.
            this.setting = setChanges...
                ( this.setting, setting, upper( mfilename ) );
        end
        function init( this, gpu )
            if gpu,
                this.net = vl_simplenn_move...
                    ( this.net, 'gpu' );
            end;
            idx = strfind( this.net.layers{end}.type, 'loss' );
            if ~isempty( idx ),
                this.net.layers{ end }.type( idx : end ) = [  ];
            end;
        end;
        function desc = iid2desc( this, iid )
            im = imread( this.db.iid2impath{ iid } );
            desc = this.im2desc( im );
        end
        function desc = im2desc( this, im )
            augmentationType = this.setting.augmentationType;
            averageImage = this.net.normalization.averageImage;
            keepAspect = this.net.normalization.keepAspect;
            interpolation = this.net.normalization.interpolation;
            layerId = this.setting.layerId;
            useGpu = isa( this.net.layers{ 1 }.weights{ 1 }, 'gpuArray' );
            switch augmentationType,
                case 'NONE',
                    numAugmentation = 1;
                case 'F5',
                    numAugmentation = 10;
                case 'F25',
                    numAugmentation = 50;
            end;
            ims = augmentImages( ...
                im, ...
                augmentationType, ...
                numAugmentation, ...
                interpolation, ...
                round( size( averageImage, 1 ) * ( 256 / 224 ) ), ...
                size( averageImage, 3 ), ...
                size( averageImage, 1 ), ...
                size( averageImage, 3 ), ...
                keepAspect );
            ims = single( ims );
            ims = bsxfun( @minus, ims, averageImage );
            if useGpu, ims = gpuArray( ims ); end;
            desc = my_simplenn( ...
                this.net, ims, [  ], [  ], ...
                'accumulate', false, ...
                'disableDropout', true, ...
                'conserveMemory', true, ...
                'backPropDepth', +inf, ...
                'targetLayerId', layerId, ...
                'sync', true ); clear ims;
            desc = gather( desc( layerId + 1 ).x ); clear res;
            desc = mean( desc, 4 );
            desc = desc( : );
        end
        % Functions for object identification.
        function name = getName( this )
            name = sprintf( 'ND_%s_OF_%s', ...
                this.setting.changes, ...
                this.net.name );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
    end
end