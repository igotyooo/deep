clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.db                                          = path.db.voc2007;
setting.io.tsDb.selectClassName                     = 'person';
setting.io.tsDb.stride                              = 32;
setting.io.tsDb.dstSide                             = 227;

setting.io.tsDb.numScale                            = 8;
setting.io.tsDb.scaleStep                           = 2;
setting.io.tsDb.startHorzScale                      = 1.5;
setting.io.tsDb.horzScaleStep                       = 0.5;
setting.io.tsDb.endHorzScale                        = 4;
setting.io.tsDb.startVertScale                      = 1.5;
setting.io.tsDb.vertScaleStep                       = 0.5;
setting.io.tsDb.endVertScale                        = 2;

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

db = Db( setting.db, path.dstDir );
db.genDb;


%% 
% Fix height. Control width to normalize aspect rate.
close all;
cname = 'aeroplane';
cid = find( cellfun( @( x )strcmp( cname, x ), db.cid2name ) );
oid2true = db.oid2cid == cid;
oid2h = ( db.oid2bbox( 3, : ) - db.oid2bbox( 1, : ) + 1 )';
oid2w = ( db.oid2bbox( 4, : ) - db.oid2bbox( 2, : ) + 1 )';

numAspect = 10;
confidence = 0.97;
resolution = 1000;
oid2how = oid2h ./ oid2w;
idx2how = logspace( -2, 2, resolution );
idx2pm = histc( oid2how( oid2true ), idx2how );
idx2cm = cumsum( idx2pm );
for c = 0 : 1 : max( idx2pm )
    if sum( idx2pm( idx2pm >= c ) ) / sum( idx2pm ) < confidence,
        indice = find( idx2pm >= c );
        mincm = idx2cm( min( indice ) );
        maxcm = idx2cm( max( indice ) );
        break;
    end
end;
aid2cm = ( mincm : ( maxcm - mincm ) / ( numAspect - 1 ) : maxcm )';
[ ~, aid2idx ] = min( abs( bsxfun( @minus, aid2cm, repmat( idx2cm', numel( aid2cm ), 1 ) ) ), [], 2 );
aid2idx = unique( aid2idx );
aid2how = idx2how( aid2idx );


numScale = 8;
confidence = 0.97;
resolution = 1000;
docScaleMag = 2;
oid2imside = min( db.iid2size( :, db.oid2iid ), [  ], 1 )';
oid2boxside = max( [ oid2h, oid2w ], [  ], 2 );
oid2iob = oid2imside ./ oid2boxside;



























