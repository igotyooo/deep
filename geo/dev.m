%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.db = path.db.voc2007;
db = Db( setting.db, path.dstDir );
db.genDb;

%% 
clc; close all; clearvars -except db path setting;

iid = 1; im = imread( db.iid2impath{ iid } );

imSize = size( im );
