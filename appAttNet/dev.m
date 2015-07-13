%% DETERMINE ASPECT RATES.
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.db = path.db.voc2007;
setting.io.tsDb.selectClassName = 'person';
setting.io.tsDb.stride = 32;
setting.io.tsDb.dstSide = 227;
db = Db( setting.db, path.dstDir );
db.genDb;

cname = 'person';
cid = find( cellfun( @( x )strcmp( cname, x ), db.cid2name ) );
numAspect = 10;
confidence = 0.97;
oid2bbox = db.oid2bbox( :, db.oid2cid == cid );
aid2how = determineAspectRates( oid2bbox, numAspect, confidence );

%% DETERMINE SCALES. (In progress.)
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.db = path.db.voc2007;
setting.io.tsDb.selectClassName = 'person';
setting.io.tsDb.stride = 32;
setting.io.tsDb.dstSide = 227;
db = Db( setting.db, path.dstDir );
db.genDb;

cname = 'person';
cid = find( cellfun( @( x )strcmp( cname, x ), db.cid2name ) );
numScale = 8;
confidence = 0.97;
resolution = 1000;
docScaleMag = 4;
[ oid2imside, oid2sideidx ] = min( db.iid2size( :, db.oid2iid ), [  ], 1 );
oid2imside = oid2imside' * sqrt( docScaleMag );
oid2sideidx = ( ( ( 1 : numel( oid2sideidx ) ) - 1 ) * 2 + oid2sideidx )';
oid2boxside = [ oid2h'; oid2w'; ];
oid2boxside = oid2boxside( oid2sideidx );
oid2iob = oid2imside ./ oid2boxside;

idx2iob = logspace( 0, 3, resolution );
idx2pm = histc( oid2iob( oid2true ), idx2iob );
idx2cm = cumsum( idx2pm );
for c = 0 : 1 : max( idx2pm )
    if sum( idx2pm( idx2pm >= c ) ) / sum( idx2pm ) < confidence,
        indice = find( idx2pm >= c );
        mincm = idx2cm( min( indice ) );
        maxcm = idx2cm( max( indice ) );
        break;
    end
end;
sid2cm = ( mincm : ( maxcm - mincm ) / ( numAspect - 1 ) : maxcm )';
[ ~, sid2idx ] = min( abs( bsxfun( @minus, sid2cm, repmat( idx2cm', numel( sid2cm ), 1 ) ) ), [], 2 );
sid2idx = unique( sid2idx );
sid2iob = idx2iob( sid2idx );

%% EXTRACT MULTI-SCALE MULTI-ASPECT DENSE REGIONS.
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.db = path.db.voc2007;
setting.io.tsDb.selectClassName = 'person';
setting.io.tsDb.stride = 32;
setting.io.tsDb.dstSide = 227;
db = Db( setting.db, path.dstDir );
db.genDb;

iid = 2;
docScaleMag = 4;
imSize = db.iid2size( :, iid );
[ imSizeMag, imTlbr, ~ ] = scaleDocument( imSize, [ 1; 1; imSize; ], 18 );
im = zeros( imSizeMag( 1 ), imSizeMag( 2 ), 3, 'uint8' );
im( imTlbr( 1 ) : imTlbr( 3 ), imTlbr( 2 ) : imTlbr( 4 ), : ) = imread( db.iid2impath{ iid } );
rid2tlbr = extMultiScaleDenseRegions...
    ( imSize, docScaleMag, 32, 227, 2.^( 0 : 0.5 : 2 ), 0.5 : 0.5 : 1.5 );
rid2tlbr( 1 : 4, : ) = bsxfun( @plus, rid2tlbr( 1 : 4, : ), repmat( imTlbr( 1 : 2 ), 2, 1 ) ) - 1;
plottlbr( rid2tlbr, im, true, 'c' );





%% MY IMAGE CROPPER.
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.db = path.db.voc2007;
setting.io.tsDb.selectClassName = 'person';
setting.io.tsDb.stride = 32;
setting.io.tsDb.dstSide = 227;
db = Db( setting.db, path.dstDir );
db.genDb;

iid = 2;
im = imread( db.iid2impath{ iid } );
imSize = db.iid2size( :, iid );
bgd = 128 * ones( 1, 1, 3, 'uint8' );
im_ = myimcrop( im, imSize, [ 1; 1; imSize; ], bgd );
imshow( im_ );


%% DISP FGD-OBJ.
clc; close all; fclose all; clear all; 
reset( gpuDevice( 2 ) ); 
addpath( genpath( '..' ) ); init;
setting.db                                          = path.db.voc2007;
setting.gpus                                        = 2;
setting.io.tsDb.selectClassName                     = 'person';
setting.io.tsDb.stride                              = 32;
setting.io.tsDb.dstSide                             = 227;
setting.io.tsDb.numScale                            = 10;
setting.io.tsDb.scaleStep                           = 2;
setting.io.tsDb.numAspect                           = 16;
setting.io.tsDb.docScaleMag                         = 4;
setting.io.tsDb.confidence                          = 0.97;
setting.io.tsDb.insectOverFgdObj                    = 0.5;
setting.io.tsDb.insectOverFgdObjForMajority         = 0.1;
setting.io.tsDb.fgdObjMajority                      = 1.5;
setting.io.tsDb.insectOverBgdObj                    = 0.2;
setting.io.tsDb.insectOverBgdRegn                   = 0.5;
setting.io.tsDb.insectOverBgdRegnForReject          = 0.9;
setting.io.tsDb.numDirection                        = 3;
setting.io.tsDb.numMaxBgdRegnPerScale               = 100;
setting.io.tsDb.stopSignError                       = 5;
setting.io.tsDb.minObjScale                         = 1 / sqrt( 2 );
setting.io.tsDb.numErode                            = 5;
setting.io.tsNet.pretrainedNetName                  = path.net.vgg_m_2048.name;
setting.io.tsNet.suppressPretrainedLayerLearnRate   = 1 / 10;
setting.io.tsNet.outFilterDepth                     = 2048;
setting.io.general.dstSide                          = 227;
setting.io.general.dstCh                            = 3;
setting.io.general.batchSize                        = 128 * numel( setting.gpus );

db = Db( setting.db, path.dstDir );
db.genDb;
io = InOutDetSingleCls( db, ...
    setting.io.tsDb, ...
    setting.io.tsNet, ...
    setting.io.general );
io.init;
%%
close all;
field = 'oid2dpid2regnsNsqr';
oid=randsample(numel(io.tsDb.tr.fgdObj.oid2iid),1);
dpid=randsample(size(io.tsDb.tr.dpid2dids,2),1);
iid=io.tsDb.tr.fgdObj.oid2iid(oid);
impath=io.tsDb.tr.iid2impath{iid};
im=imread(impath);
tlbr=io.tsDb.tr.fgdObj.( field ){oid,dpid};
tlbr=tlbr(:,tlbr(end,:)==1);
if ~isempty(tlbr),
    tlbr=tlbr(:,randsample(size(tlbr,2),1));
    im=myimcrop(im,size(im),tlbr,128*ones(1,1,3));
    imshow(imresize(im,[500,500]));
    title(mat2str(io.tsDb.tr.dpid2dids(:,dpid)'));
end;

%% DISP BGD-OBJ.
clc; close all; fclose all; clear all; 
reset( gpuDevice( 2 ) ); 
addpath( genpath( '..' ) ); init;
setting.db                                          = path.db.voc2007;
setting.gpus                                        = 2;
setting.io.tsDb.selectClassName                     = 'person';
setting.io.tsDb.stride                              = 32;
setting.io.tsDb.dstSide                             = 227;
setting.io.tsDb.numScale                            = 10;
setting.io.tsDb.scaleStep                           = 2;
setting.io.tsDb.numAspect                           = 16;
setting.io.tsDb.docScaleMag                         = 4;
setting.io.tsDb.confidence                          = 0.97;
setting.io.tsDb.insectOverFgdObj                    = 0.5;
setting.io.tsDb.insectOverFgdObjForMajority         = 0.1;
setting.io.tsDb.fgdObjMajority                      = 1.5;
setting.io.tsDb.insectOverBgdObj                    = 0.2;
setting.io.tsDb.insectOverBgdRegn                   = 0.5;
setting.io.tsDb.insectOverBgdRegnForReject          = 0.9;
setting.io.tsDb.numDirection                        = 3;
setting.io.tsDb.numMaxBgdRegnPerScale               = 100;
setting.io.tsDb.stopSignError                       = 5;
setting.io.tsDb.minObjScale                         = 1 / sqrt( 2 );
setting.io.tsDb.numErode                            = 5;
setting.io.tsNet.pretrainedNetName                  = path.net.vgg_m_2048.name;
setting.io.tsNet.suppressPretrainedLayerLearnRate   = 1 / 10;
setting.io.tsNet.outFilterDepth                     = 2048;
setting.io.general.dstSide                          = 227;
setting.io.general.dstCh                            = 3;
setting.io.general.batchSize                        = 128 * numel( setting.gpus );

db = Db( setting.db, path.dstDir );
db.genDb;
io = InOutDetSingleCls( db, ...
    setting.io.tsDb, ...
    setting.io.tsNet, ...
    setting.io.general );
io.init;

close all;
field = 'oid2regnsNsqr';
oid=randsample(numel(io.tsDb.tr.bgdObj.oid2iid),1);
iid=io.tsDb.tr.bgdObj.oid2iid(oid);
impath=io.tsDb.tr.iid2impath{iid};
im=imread(impath);
tlbr=io.tsDb.tr.bgdObj.( field ){ oid };
if ~isempty(tlbr),
    tlbr=tlbr(:,randsample(size(tlbr,2),1));
    im=myimcrop(im,size(im),tlbr,128*ones(1,1,3));
    imshow(imresize(im,[500,500]));
    title('bgd obj');
end;


%% DISP BGD.
clc; close all; fclose all; clear all; 
reset( gpuDevice( 2 ) ); 
addpath( genpath( '..' ) ); init;
setting.db                                          = path.db.voc2007;
setting.gpus                                        = 2;
setting.io.tsDb.selectClassName                     = 'person';
setting.io.tsDb.stride                              = 32;
setting.io.tsDb.dstSide                             = 227;
setting.io.tsDb.numScale                            = 10;
setting.io.tsDb.scaleStep                           = 2;
setting.io.tsDb.numAspect                           = 16;
setting.io.tsDb.docScaleMag                         = 4;
setting.io.tsDb.confidence                          = 0.97;
setting.io.tsDb.insectOverFgdObj                    = 0.5;
setting.io.tsDb.insectOverFgdObjForMajority         = 0.1;
setting.io.tsDb.fgdObjMajority                      = 1.5;
setting.io.tsDb.insectOverBgdObj                    = 0.2;
setting.io.tsDb.insectOverBgdRegn                   = 0.5;
setting.io.tsDb.insectOverBgdRegnForReject          = 0.9;
setting.io.tsDb.numDirection                        = 3;
setting.io.tsDb.numMaxBgdRegnPerScale               = 100;
setting.io.tsDb.stopSignError                       = 5;
setting.io.tsDb.minObjScale                         = 1 / sqrt( 2 );
setting.io.tsDb.numErode                            = 5;
setting.io.tsNet.pretrainedNetName                  = path.net.vgg_m_2048.name;
setting.io.tsNet.suppressPretrainedLayerLearnRate   = 1 / 10;
setting.io.tsNet.outFilterDepth                     = 2048;
setting.io.general.dstSide                          = 227;
setting.io.general.dstCh                            = 3;
setting.io.general.batchSize                        = 128 * numel( setting.gpus );

db = Db( setting.db, path.dstDir );
db.genDb;
io = InOutDetSingleCls( db, ...
    setting.io.tsDb, ...
    setting.io.tsNet, ...
    setting.io.general );
io.init;

close all;
field = 'iid2sid2regnsNsqr';
iid=randsample(size(io.tsDb.tr.bgd.(field),1),1);
s=randsample(size(io.tsDb.tr.bgd.(field),2),1);
impath=io.tsDb.tr.iid2impath{iid};
im=imread(impath);
tlbr=io.tsDb.tr.bgd.(field){iid,s};
if ~isempty(tlbr),
    tlbr=tlbr(:,randsample(size(tlbr,2),1));
    im=myimcrop(im,size(im),tlbr,128*ones(1,1,3));
    imshow(imresize(im,[500,500]));
    title('bgd');
end;

%% DISP NET INPUT DIRECTLY.
clc; close all; fclose all; clear all; 
reset( gpuDevice( 2 ) ); 
addpath( genpath( '..' ) ); init;
setting.db                                          = path.db.voc2007;
setting.gpus                                        = 2;
setting.io.tsDb.selectClassName                     = 'person';
setting.io.tsDb.stride                              = 32;
setting.io.tsDb.dstSide                             = 227;
setting.io.tsDb.numScale                            = 10;
setting.io.tsDb.scaleStep                           = 2;
setting.io.tsDb.numAspect                           = 16;
setting.io.tsDb.docScaleMag                         = 4;
setting.io.tsDb.confidence                          = 0.97;
setting.io.tsDb.insectOverFgdObj                    = 0.5;
setting.io.tsDb.insectOverFgdObjForMajority         = 0.1;
setting.io.tsDb.fgdObjMajority                      = 1.5;
setting.io.tsDb.insectOverBgdObj                    = 0.2;
setting.io.tsDb.insectOverBgdRegn                   = 0.5;
setting.io.tsDb.insectOverBgdRegnForReject          = 0.9;
setting.io.tsDb.numDirection                        = 3;
setting.io.tsDb.numMaxBgdRegnPerScale               = 100;
setting.io.tsDb.stopSignError                       = 5;
setting.io.tsDb.minObjScale                         = 1 / sqrt( 2 );
setting.io.tsDb.numErode                            = 5;
setting.io.tsNet.pretrainedNetName                  = path.net.vgg_m_2048.name;
setting.io.tsNet.suppressPretrainedLayerLearnRate   = 1 / 10;
setting.io.tsNet.outFilterDepth                     = 2048;
setting.io.general.dstSide                          = 227;
setting.io.general.dstCh                            = 3;
setting.io.general.batchSize                        = 128 * numel( setting.gpus );

db = Db( setting.db, path.dstDir );
db.genDb;
io = InOutDetSingleCls( db, ...
    setting.io.tsDb, ...
    setting.io.tsNet, ...
    setting.io.general );
io.init;

[ ims, gts ] = io.provdBchVal;
for i = 1 : size( ims, 4 );
    gt = gts( :, :, :, i );
    gt = gt( : );
    im = ims( :, :, :, i );
    im = bsxfun( @plus, im, io.rgbMean );
    imshow( uint8( im ) );
    title( mat2str( gt' ) );
    waitforbuttonpress;
end;





%%
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.db = path.db.voc2007;
setting.io.tsDb.selectClassName = 'person';
setting.io.tsDb.stride = 32;
setting.io.tsDb.dstSide = 227;
db = Db( setting.db, path.dstDir );
db.genDb;

%% 
clearvars -except db path setting;
clc; close all;
clsName = 'aeroplane';
cid = find( cellfun( @( name )strcmp( name, clsName ), db.cid2name ) );
iids = unique( db.oid2iid( db.oid2cid == cid ) );
iid = iids( 3 );
im = imread( db.iid2impath{ iid } );
imshow( im );

numAspect = 5;
confidence = 0.97;
numScale = 6;
scaleStep = 2;
patchMargin = 0.5;
docScaleMag = 4;
itpltn = 'bicubic';

rgbMean = single( zeros( 1, 1, 3 ) );
aspects = determineAspectRates( db.oid2bbox( :, db.oid2cid == cid ), numAspect - 1, confidence );
aspects = unique( cat( 1, 1, aspects ) );
scales = scaleStep .^ ( 0 : 0.5 : ( 0.5 * ( numScale - 1 ) ) );
patchSize = 219;
stride = 32;
net = 0;

lyid = numel( this.attNet.layers ) - 1;
imSize = size( im ); imSize = imSize( 1 : 2 ); imSize = imSize( : );
pside0 = min( imSize ) / patchMargin;
cnt = 0;
rid2tlbr = cell( numScale * numAspect, 1 );
rid2out = cell( numScale * numAspect, 1 );
for s = 1 : numScale,
    pside = pside0 / scales( s );
    for a = 1 : numAspect,
        psider = pside;
        psidec = pside / aspects( a );
        mar2im = round( [ psider; psidec ] * patchMargin );
        bnd = [ 1 - mar2im; imSize + mar2im ];
        srcr = bnd( 3 ) - bnd( 1 ) + 1;
        srcc = bnd( 4 ) - bnd( 2 ) + 1;
        dstr = patchSize * srcr / psider;
        dstc = patchSize * srcc / psidec;
        if dstr * dstc > 2859 * 5448,
            fprintf( '%s: Warning) Im of s%d a%d rejected.\n', ...
                upper( mfilename ), s, a ); continue;
        end;
        im_ = cropAndNormalizeIm( single( im ), imSize, bnd, rgbMean );
        im_ = imresize( im_, [ dstr, dstc ], 'method', itpltn ); 
        % Feed-foreward.
        if isa( net.layers{ 1 }.weights{ 1 }, 'gpuArray' ), im_ = gpuArray( im_ ); end;
        res = my_simplenn( ...
            net, im_, [  ], [  ], ...
            'accumulate', false, ...
            'disableDropout', true, ...
            'conserveMemory', true, ...
            'backPropDepth', +inf, ...
            'targetLayerId', lyid, ...
            'sync', true ); clear im_;
        % Form activations.
        outs = gather( res( lid + 1 ).x );
        [ nr, nc, z ] = size( outs );
        outs = reshape( permute( outs, [ 3, 1, 2 ] ), z, nr * nc );
        % Form geometries.
        r = ( ( 1 : nr ) - 1 ) * stride + 1;
        c = ( ( 1 : nc ) - 1 ) * stride + 1;
        [ c, r ] = meshgrid( c, r );
        regns = cat( 3, r, c );
        regns = cat( 3, regns, regns + stride - 1 );
        regns = reshape( permute( regns, [ 3, 1, 2 ] ), 4, nr * nc );
        regns = cat( 1, regns, s * ones( 1, nr * nc  ), a * ones( 1, nr * nc  ) );
        % Back projection.
        regns = resizeTlbr( regns, [ dstr; dstc; ], [ srcr; srcc; ] );
        regns = bsxfun( @minus, regns, [ mar2im; mar2im; ] );
        cnt = cnt + 1;
        rid2tlbr{ cnt } = regns;
        rid2out{ cnt } = outs;
        % plottlbr( [ 1; 1; patchSize; patchSize; ], uint8( im_ ), false, 'c' );
        % waitforbuttonpress; 
    end;
end;
rid2tlbr = cat( 2, rid2tlbr{ : } );
mar2im = min( rid2tlbr( 1 : 2, : ), [  ], 2 ) - 1;
imGlobal = cropAndNormalizeIm( single( im ), imSize, [ 1 - mar2im; imSize + mar2im ], rgbMean );



% imSize = size( im ); imSize = imSize( 1 : 2 ); imSize = imSize( : );
% mag2im = ceil( imSize / 2 );
% imSizeMag = imSize + mag2im * 2;
% numAspect = numel( aspects );
% marginr = 0;
% marginc = 0;
% for a = 1 : numAspect,
%     rside = min( imSizeMag );
%     cside = rside / aspects( a );
%     marginr_ = rside - imSizeMag( 1 );
%     marginc_ = cside - imSizeMag( 2 );
%     if marginr_ > marginr, marginr = marginr_; end;
%     if marginc_ > marginc, marginc = marginc_; end;
% end;
% newmag2mag = ceil( [ marginr; marginc; ] / 2 );
% global2im = newmag2mag + mag2im;
% imGlobal = cropAndNormalizeIm( single( im ), imSize, [ 1 - global2im; imSize + global2im ], rgbMean );