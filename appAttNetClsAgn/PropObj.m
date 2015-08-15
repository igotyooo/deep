classdef PropObj < handle
    properties
        db;
        propNet;
        stride;
        patchSide;
        scales;
        settingProp0;
        settingProp;
    end
    methods( Access = public )
        function this = PropObj( db, propNet, settingProp0 )
            this.db = db;
            this.propNet = propNet;
            this.settingProp0.numScaling = 24;
            this.settingProp0.dilate = 1 / 4;
            this.settingProp0.posIntOverRegnMoreThan = 1 / 3;
            this.settingProp0 = setChanges...
                ( this.settingProp0, settingProp0, upper( mfilename ) );
        end
        function init( this, gpus )
            % Set parameters.
            numScaling = this.settingProp0.numScaling;
            posIntOverRegnMoreThan = this.settingProp0.posIntOverRegnMoreThan;
            % Fetch net on GPU.
            this.propNet.layers{ end }.type = 'softmax';
            this.propNet = Net.fetchNetOnGpu( this.propNet, gpus );
            % Determine stride and patch side.
            fprintf( '%s: Determine stride and patch side.\n', ...
                upper( mfilename ) );
            [ this.patchSide, this.stride ] = ...
                getNetProperties( this.propNet, numel( this.propNet.layers ) - 1 );
            % Determine scaling factors.
            fprintf( '%s: Determine scaling factors.\n', ...
                upper( mfilename ) );
            oid2tlbr = this.db.oid2bbox( :, this.db.iid2setid( this.db.oid2iid ) == 1 );
            referenceSide = this.patchSide * sqrt( posIntOverRegnMoreThan );
            [ scalesRow, scalesCol ] = determineImageScaling...
                ( oid2tlbr, numScaling, referenceSide, true );
            this.scales = [ scalesRow, scalesCol ]';
            fprintf( '%s: Done.\n', ...
                upper( mfilename ) );
        end
        function propObj( this )
            iids = this.db.getTeiids;
            fprintf( '%s: Check if proposals exist.\n', ...
                upper( mfilename ) );
            paths = arrayfun( ...
                @( iid )this.getPath( iid ), iids, ...
                'UniformOutput', false );
            exists = cellfun( ...
                @( path )exist( path, 'file' ), paths );
            this.makeDir;
            iids = iids( ~exists );
            numIm = numel( iids );
            cummt = 0;
            for iidx = 1 : numIm; itime = tic;
                iid = iids( iidx );
                this.iid2prop( iid );
                cummt = cummt + toc( itime );
                fprintf( '%s: ', upper( mfilename ) );
                disploop( numIm, iidx, ...
                    'Prop obj.', cummt );
            end;
        end
        function [ rid2tlbr, rid2out ] = ...
                iid2prop( this, iid )
            % Initial guess.
            fpath = this.getPath( iid );
            try
                data = load( fpath );
                rid2tlbr = data.det.rid2tlbr;
                rid2out = data.det.rid2out;
            catch
                im = imread( this.db.iid2impath{ iid } );
                [ rid2tlbr, rid2out ] = ...
                    this.im2prop0( im );
                prop.rid2tlbr = rid2tlbr;
                prop.rid2out = rid2out;
                save( fpath, 'prop' );
            end;
            if nargout,
                % % Scale/aspect selection.
                % numScale = this.settingProp0.numScale;
                % numAspect = this.settingProp0.numAspect;
                % sids = this.settingProp.selectScaleIds;
                % aids = this.settingProp.selectAspectIds;
                % if numScale ~= numel( sids ),
                %     did2ok = ismember( rid2tlbr( 5, : ), sids );
                %     did2det = did2det( :, did2ok );
                %     did2score = did2score( :, did2ok );
                % end;
                % if numAspect ~= numel( aids ),
                %     did2ok = ismember( did2det( 6, : ), aids );
                %     did2det = did2det( :, did2ok );
                %     did2score = did2score( :, did2ok );
                % end;
                % % Compute each region score here.
                % % NMS.
                % overlap = this.settingInitMrg.overlap;
                % [ did2det, did2score, ~ ] = ...
                %     nms( [ did2det; did2score ]', overlap );
                % did2det = did2det';
            end;
        end
        function [ rid2out, rid2tlbr ] = im2prop0( this, im )
            dilate = this.settingProp0.dilate;
            [ r, c, ~ ] = size( im );
            imSize0 = [ r; c; ];
            sid2size = round( bsxfun( @times, this.scales, imSize0 ) );
            rid2out = ...
                extractDenseActivations( ...
                im, ...
                this.propNet, ...
                numel( this.propNet.layers ), ...
                sid2size, ...
                this.patchSide, ...
                dilate );
            rid2tlbr = ...
                extractDenseRegions( ...
                imSize0, ...
                sid2size, ...
                this.patchSide, ...
                this.stride, ...
                dilate );
            if size( rid2out, 2 ) ~= size( rid2tlbr, 2 ),
                error( 'Inconsistent number of regions.\n' ); end;
        end
        % Functions for identification.
        function name = getName( this )
            name = sprintf( ...
                'PROP_%s_OF_%s', ...
                this.settingProp0.changes, ...
                this.propNet.getNetName );
            name( strfind( name, '__' ) ) = '';
            if name( end ) == '_', name( end ) = ''; end;
        end
        function dir = getDir( this )
            name = this.getName;
            if length( name ) > 150, 
                name = sum( ( name - 0 ) .* ( 1 : numel( name ) ) ); 
                name = sprintf( '%010d', name );
                name = strcat( 'PROP_', name );
            end
            dir = fullfile...
                ( this.db.dstDir, name );
        end
        function dir = makeDir( this )
            dir = this.getDir;
            if ~exist( dir, 'dir' ), mkdir( dir ); end;
        end
        function fpath = getPath( this, iid )
            fname = sprintf...
                ( 'ID%06d.mat', iid );
            fpath = fullfile...
                ( this.getDir, fname );
        end
    end
end

