classdef ImDscrber < handle
    properties
        srcDb;
        srcDscrber;     % Can be Fisher, NeuralCode, Sift, Gist, Hog, ...
        setting;
    end
    methods
        function this = ImDscrber...
                ( srcDb, srcDscrber, setting )
            this.srcDb = srcDb;
            this.srcDscrber = srcDscrber;
            this.setting.weights = ones( size( srcDscrber ) );
            this.setting.aug = false;
            this.setting.keepAspect = true;
            this.setting = setChanges...
                ( this.setting, setting, upper( mfilename ) );
        end
        function descDb( this )
            numDscrber = numel( this.srcDscrber );
            did2iid2exist = cell( numDscrber, 1 );
            for did = 1 : numDscrber
                fprintf( '%s: Check if descs exist.\n', ...
                    upper( mfilename ) );
                iid2vpath = cellfun( ...
                    @( iid )this.getPath( iid, did ), ...
                    num2cell( 1 : this.srcDb.getNumIm )', ...
                    'UniformOutput', false );
                did2iid2exist{ did } = cellfun( ...
                    @( path )exist( path, 'file' ), ...
                    iid2vpath );
                this.makeDir( did );
            end
            did2iid2exist = cat( 2, did2iid2exist{ : } );
            iid2exist = prod( did2iid2exist, 2 );
            iids = find( ~iid2exist );
            if isempty( iids ),
                fprintf( '%s: No im to desc.\n', ...
                    upper( mfilename ) ); return;
            end;
            cnt = 0; cummt = 0; numIm = numel( iids );
            for iid = iids'; itime = tic;
                this.iid2desc( iid, 'NONE', 'NONE' );
                cummt = cummt + toc( itime ); 
                cnt = cnt + 1;
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numIm, cnt, ...
                    'Desc im.', cummt );
            end
        end
        function desc = iid2desc...
                ( this, iid, kernel, norm )
            aug = this.setting.aug;
            keepAspect = this.setting.keepAspect;
            weights = this.setting.weights;
            numDscrber = numel( this.srcDscrber );
            did2desc = cell( numDscrber, 1 );
            for did = 1 : numDscrber
                fpath = this.getPath( iid, did );
                try
                    data = load( fpath );
                    desc = data.desc;
                catch
                    im = imread( this.srcDb.iid2ifpath{ iid } );
                    % Augment image.
                    if aug
                        [ h, w, ~ ] = size( im );
                        rate = 224 / 256;
                        if keepAspect
                            dsth = round( h * rate );
                            dstw = round( w * rate );
                        else
                            dsth = round( min( h, w ) * rate );
                            dstw = dsth;
                        end
                        ims = this.augIms( im, dsth, dstw );
                    else
                        ims = { im };
                    end
                    numaug = numel( ims );
                    desc = cell( numaug, 1 );
                    for aid = 1 : numaug;
                        im = ims{ aid };
                        desc{ aid } = this.srcDscrber{ did }.im2desc( im );
                    end
                    desc = cat( 2, desc{ : } );
                    save( fpath, 'desc' );
                end
                desc = kernelMap( desc, kernel );
                desc = nmlzVecs( desc, norm );
                did2desc{ did } = desc * weights( did );
            end
            if nargout, desc = cat( 1, did2desc{ : } ); end;
        end
        function desc = ...
                im2desc( this, im, kernel, norm )
            aug = this.setting.aug;
            keepAspect = this.setting.keepAspect;
            weights = this.setting.weights;
            numDscrber = numel( this.srcDscrber );
            did2desc = cell( numDscrber, 1 );
            for did = 1 : numDscrber,
                if aug
                    [ h, w, ~ ] = size( im );
                    rate = dstSideBacktup / srcSideBacktup;
                    if keepAspect
                        dsth = round( min( h, w ) * rate );
                        dstw = dsth;
                    else
                        dsth = round( h * rate );
                        dstw = round( w * rate );
                    end
                    ims = this.augIms( im, dsth, dstw );
                else
                    ims = { im };
                end
                numaug = numel( ims );
                desc = cell( numaug, 1 );
                for aid = 1 : numaug;
                    im = ims{ aid };
                    desc{ aid } = this.srcDscrber{ did }.im2desc( im );
                end
                desc = cat( 2, desc{ : } );
                desc = kernelMap( desc, kernel );
                desc = nmlzVecs( desc, norm );
                did2desc{ did } = desc * weights( did );
            end
            desc = cat( 1, did2desc{ : } );
        end
        % Functions for data I/O.
        function name = getName( this, dscrberId )
            if nargin > 1
                if ~isempty( this.setting.changes )
                    changes = strsplit( this.setting.changes, '_' );
                    foo = cellfun( @( str )strcmp( str( 1 : 2 ), 'W[' ), changes );
                    if any( foo ),
                        len = numel( changes{ foo } );
                        idx = strfind( this.setting.changes, changes{ foo } );
                        changes = this.setting.changes;
                        changes( idx : idx + len - 1 ) = [];
                    else
                        changes = this.setting.changes;
                    end
                else
                    changes = this.setting.changes;
                end
                name = sprintf( 'ID_%s_OF_%s', ...
                    changes, ...
                    this.srcDscrber{ dscrberId }.getName );
            else
                numDscrber = numel( this.srcDscrber );
                name = {  };
                name{ end + 1 } = ...
                    sprintf( 'ID_%s_OF_', this.setting.changes );
                for did = 1 : numDscrber
                    if did > 1, name{ end + 1 } = '_AND_'; end;
                    name{ end + 1 } = ...
                        this.srcDscrber{ did }.getName;
                end
                name = cat( 2, name{ : } );
            end
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getDir( this, dscrberId )
            name = this.getName( dscrberId );
            if length( name ) > 150, 
                name = sum( ( name - 0 ) .* ( 1 : numel( name ) ) ); 
                name = sprintf( '%010d', name );
                name = strcat( 'ID_', name );
            end
            dir = fullfile...
                ( this.srcDb.dir, name );
        end
        function dir = makeDir( this, dscrberId )
            dir = this.getDir( dscrberId );
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function fpath = getPath( this, iid, dscrberId )
            fname = sprintf...
                ( 'ID%06d.mat', iid );
            fpath = fullfile...
                ( this.getDir( dscrberId ), fname );
        end
    end
    methods( Static )
        function ims = augIms( im, dsth, dstw )
            aid2tf = [ ...
                0.5, 0.0, 0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 1.0, 1.0;
                0.5, 0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.0, 1.0;
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0; ];
            numAug = size( aid2tf, 2 );
            [ h, w, ~ ] = size( im );
            ims = cell( numAug, 1 );
            for aid = 1 : numAug
                tf = aid2tf( :, aid );
                % Crop.
                dx = floor( ( w - dstw ) * tf( 2 ) ); % Orientation of x-axis. ( = bias of x )
                dy = floor( ( h - dsth ) * tf( 1 ) ); % Orientation of y-axis. ( = bias of y )
                sx = ( 1 : dstw ) + dx;
                sy = ( 1 : dsth ) + dy;
                % Flip.
                if tf( 3 ), sx = fliplr( sx ); end
                im_ = im( sy, sx, : );
                ims{ aid } = im_;
            end
        end
    end
end

