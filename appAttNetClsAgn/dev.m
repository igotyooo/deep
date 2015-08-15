% 1. Remaining issues to define postive regions.
%       1) A sub-object should be truncated enough (even if it occupies large area).
%          e.g. a small cat (target) stting on a sofa (sub-object).
%       2) A sub-object should occupy a small enough area.
%          e.g. a person (target) holing a bottle (sub-object).
%       Is these really neccessary? 
%       Why should we reject a small cat sitting on a small chair?
%       Why should we reject a small chair where a small cat is sitting on?
%       Then, if we ignore these constraints, how can we handle two objects with similar sizes?
%       -> Simply set a majority factor. 
%       (i.e. fully-included target object size / "fully"-included sub-object size > 1.5)
%       This majority factor should not be too large to prevent rejecting a small cat setting on a small chair.
% 2. Remaining issues to define semi-negative regions.
%       1) All fully-included object are small enough.
%       2) All fully included objects have similar sizes.
%       Is these really neccessary? 

clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.db = path.db.voc2007;
db = Db( setting.db, path.dstDir );
db.genDb;

%% 
clc; clearvars -except db path setting; close all; 

rng( 'shuffle' );
iid = 9374; 446; randsample( db.getNumIm, 1 ); 4203; 1297; 3595; 
im = imread( db.iid2impath{ iid } );
imSize = size( im )'; imSize = imSize( 1 : 2 );
oid2tlbr = db.oid2bbox( :, db.oid2iid == iid );
numObj = size( oid2tlbr, 2 );

% Parameters for dense region extraction.
stride = 32;
patchSide = 227;
numScale = 16;
numAspect = 16;
confidence = 0.97;
violate = 1 / 4;
% Parameters for positive mining.
posIntOverRegnMoreThan = 1 / 3;         % A target object should be large enough.
posIntOverTarObjMoreThan = 0.99;        % A target object should be fully-included.
posIntOverSubObjMoreThan = 0.6;         % A sub-object should be majorly-included.
posIntMajorityMoreThan = 2;             % A target object should large enough w.r.t. the sub-objects.
% Parameters for semi-negative mining.
snegIntOverRegnMoreThan = 1 / 36;       % A sub-object should not be too small.
snegIntOverObjMoreThan = 0.3;           % The region is truncating the target object.
snegIntOverObjLessThan = 0.7;           % The region is truncating the target object.
% Parameters for negative mining.
negIntOverObjLessThan = 0.1;            % Very small overlap is allowed for background region.
% Compute bisic informations.
referenceSide = patchSide * sqrt( posIntOverRegnMoreThan );
sid2s = determineScales...
    ( db.oid2bbox, referenceSide, numScale, confidence );
aid2a = determineAspectRates...
    ( db.oid2bbox, numAspect, confidence );

rid2tlbr = extMultiScaleDenseRegions2...
    ( imSize, stride, patchSide, sid2s, aid2a, violate );
rid2rect = tlbr2rect( rid2tlbr );
rid2area = prod( rid2rect( 3 : 4, : ), 1 )';
oid2rect = tlbr2rect( oid2tlbr );
oid2area = prod( oid2rect( 3 : 4, : ), 1 )';
rid2oid2int = rectint( rid2rect', oid2rect' );
rid2oid2ioo = bsxfun( @times, rid2oid2int, 1 ./ oid2area' );
rid2oid2ior = bsxfun( @times, rid2oid2int, 1 ./ rid2area );
% Positive mining.
% 1. A target object should be fully-included.
% 2. A target object should occupy a large enough area.
% 3. A target object should large enough w.r.t. the sub-objects.
rid2oid2fullinc = rid2oid2ioo > posIntOverTarObjMoreThan;
rid2oid2big = rid2oid2ior > posIntOverRegnMoreThan;
rid2oid2fullandbig = rid2oid2fullinc & rid2oid2big;
rid2tararea = max( rid2oid2int .* rid2oid2fullandbig, [  ], 2 );
rid2oid2incarea = rid2oid2int .* ( rid2oid2ioo > posIntOverSubObjMoreThan );
rid2oid2incarea = bsxfun( @times, rid2oid2incarea, 1 ./ rid2tararea );
foo = rid2oid2incarea >= ( 1 / posIntMajorityMoreThan );
rid2oid2fullandbig = rid2oid2fullandbig & foo;
rid2major = sum( foo, 2 ) == 1;
[ prid2oid, ~ ] = find( rid2oid2fullandbig( rid2major, : )' );
rid2ok = any( rid2oid2fullandbig, 2 ) & rid2major;
prid2tlbr = rid2tlbr( :, rid2ok );

posMinMargin = 0.1;
oid2mgtlbr = scaleBoxes( oid2tlbr, 1 + 2 * posMinMargin, 1 + 2 * posMinMargin );
oid2mgtlbr = round( oid2mgtlbr );
oid2pregns = cell( numObj, 1 );
for oid = 1 : numObj,
    prids = prid2oid == oid;
    if sum( prids ),
        ptlbrs = prid2tlbr( :, prids );
        ptlbrs( 1, ptlbrs( 1, : ) > oid2mgtlbr( 1, oid ) ) = oid2mgtlbr( 1, oid );
        ptlbrs( 2, ptlbrs( 2, : ) > oid2mgtlbr( 2, oid ) ) = oid2mgtlbr( 2, oid );
        ptlbrs( 3, ptlbrs( 3, : ) < oid2mgtlbr( 3, oid ) ) = oid2mgtlbr( 3, oid );
        ptlbrs( 4, ptlbrs( 4, : ) < oid2mgtlbr( 4, oid ) ) = oid2mgtlbr( 4, oid );
        oid2pregns{ oid } = ptlbrs;
    else
        oid2pregns{ oid } = oid2mgtlbr( :, oid );
    end;
end;
% Semi-negative mining.
% The region is truncating the target object.
rid2oid2issub = rid2oid2ior >= snegIntOverRegnMoreThan;
rid2issub = any( rid2oid2issub, 2 );
rid2oid2ok = snegIntOverObjMoreThan <= rid2oid2ioo & ...
    snegIntOverObjLessThan >= rid2oid2ioo;
rid2ok = rid2issub & all( eq( rid2oid2issub, rid2oid2ok ), 2 );
snrid2tlbr = rid2tlbr( :, rid2ok );
snrid2oid2ok = rid2oid2ok( rid2ok, : );
oid2snregns = cell( numObj, 1 );
for oid = 1 : numObj,
    snrids = snrid2oid2ok( :, oid );
    if sum( snrids ),
        oid2snregns{ oid } = snrid2tlbr( :, snrids );
    else
        tlbr = oid2tlbr( :, oid );
        w = floor( ( tlbr( 4 ) - tlbr( 2 ) + 1 ) / 2 );
        h = floor( ( tlbr( 3 ) - tlbr( 1 ) + 1 ) / 2 );
        snregns = repmat( tlbr, 1, 4 );
        snregns( 1, 1 ) = snregns( 1, 1 ) + h;
        snregns( 2, 2 ) = snregns( 2, 2 ) + w;
        snregns( 3, 3 ) = snregns( 3, 3 ) - h;
        snregns( 4, 4 ) = snregns( 4, 4 ) - w;
        oid2snregns{ oid } = snregns;
    end;
end;
% Negative mining.
% Very small overlap is allowed for background region.
rid2ok = all( rid2oid2ioo <= negIntOverObjLessThan, 2 );
nrid2tlbr = rid2tlbr( :, rid2ok );
sid2nregns = cell( numScale, 1 );
for sid = 1 : numScale,
    nrids = find( nrid2tlbr( 5, : ) == sid );
    nrids = randsample( nrids, min( numel( nrids ), 500 ) );
    sid2nregns{ sid } = nrid2tlbr( :, nrids );
end;


% figure( 1 ); plottlbr( oid2tlbr, im, false, 'c', cellfun( @num2str, num2cell( 1 : numObj ), 'UniformOutput', false ) );
% figure( 2 ); plottlbr( prid2tlbr, im, true, 'c', cellfun( @num2str, num2cell( prid2oid ), 'UniformOutput', false ) );
% figure( 3 ); plottlbr( snrid2tlbr, im, true, 'c' );
% figure( 4 ); plottlbr( nrid2tlbr, im, true, 'c' );

figure( 1 ); plottlbr( oid2tlbr, im, false, 'c', cellfun( @num2str, num2cell( 1 : numObj ), 'UniformOutput', false ) );
figure( 2 ); plottlbr( oid2pregns{ 1 }, im, false, 'c' );
% figure( 3 ); plottlbr( oid2snregns{ 2 }, im, true, 'c' );
% figure( 4 ); plottlbr( sid2nregns{ 3 }, im, true, 'c' );


%% DETERMINE SCALES AND ASPECTS
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.db = path.db.voc2007;
db = Db( setting.db, path.dstDir );
db.genDb;



%%
clc; close all; clearvars -except db;
oid2tlbr = db.oid2bbox;
numSize = 256;
patchSide = 227;
posIntOverRegnMoreThan = 1 / 3;
referenceSide = patchSide * sqrt( posIntOverRegnMoreThan );
[ sid2hs, sid2ws ] = determineImageScaling...
    ( oid2tlbr, numSize, referenceSide, true );























