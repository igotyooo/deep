%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.gpu                         = 1;
setting.db                          = path.db.voc2007;
setting.net                         = path.net.caffeRef;
setting.neuralDesc.layerId          = 19;
setting.neuralDesc.augmentationType = 'NONE';
setting.svm.kernel                  = 'NONE';
setting.svm.norm                    = 'L2';
if strcmp( setting.db.name, 'VOC2007' ),
setting.svm.c                       = 1;
else
setting.svm.c                       = 10;
end;
setting.svm.epsilon                 = 1e-3;
setting.svm.biasMultiplier          = 1;
setting.svm.biasLearningRate        = 0.5;
setting.svm.loss                    = 'HINGE';
setting.svm.solver                  = 'SDCA';

%% DO THE JOB.
reset( gpuDevice( setting.gpu ) );
db = Db( setting.db, path.dstDir );
db.genDb;
net = load( setting.net.path );
net.name = setting.net.name;
neuralDesc = NeuralDscrber( db, net, setting.neuralDesc );
neuralDesc.init( setting.gpu );
imDscrber = ImDscrber( db, { neuralDesc }, [  ] );
imDscrber.descDb;
svm = Svm( db, imDscrber, setting.svm );
svm.trainSvm;
svm.evalSvm( 'visionresearchreport@gmail.com' );