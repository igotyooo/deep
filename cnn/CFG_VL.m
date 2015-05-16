function cfg = CFG_VL
    % ________________________
    % 1ST CONVOLUTIONAL LAYER.
    lid = 0;
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'conv';
    cfg{ lid, 1 }.filterSize = 11;
    cfg{ lid, 1 }.filterDepth = 3;
    cfg{ lid, 1 }.numFilter = 96;
    cfg{ lid, 1 }.stride = 4;
    cfg{ lid, 1 }.pad = 0;
    cfg{ lid, 1 }.initWScal = 1;
    cfg{ lid, 1 }.initB = 0;
    cfg{ lid, 1 }.filtersLearningRate = 1;
    cfg{ lid, 1 }.biasesLearningRate = 2;
    cfg{ lid, 1 }.filtersWeightDecay = 1;       % Just 1 or 0. It means that the weight decay value is applied to filters.
    cfg{ lid, 1 }.biasesWeightDecay = 0;        % Just 1 or 0. It means that the weight decay value is not applied to bias.
	% Out dim: 55 * 55 * 96.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'relu';
    % Out dim: 55 * 55 * 96.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'pool';
    cfg{ lid, 1 }.method = 'max';
    cfg{ lid, 1 }.windowSize = 3;
    cfg{ lid, 1 }.stride = 2;
    cfg{ lid, 1 }.pad = 0;
    % Out dim: 27 * 27 * 96.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'normalize';
    cfg{ lid, 1 }.localSize = 5;
    cfg{ lid, 1 }.alpha = 0.0001;
    cfg{ lid, 1 }.beta = 0.75;
    % Out dim: 27 * 27 * 96.

    
    % ________________________
    % 2ND CONVOLUTIONAL LAYER.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'conv';
    cfg{ lid, 1 }.filterSize = 5;
    cfg{ lid, 1 }.filterDepth = 48;               % Shouldn't it be 96?
    cfg{ lid, 1 }.numFilter = 256;
    cfg{ lid, 1 }.stride = 1;
    cfg{ lid, 1 }.pad = 2;
    cfg{ lid, 1 }.initWScal = 1;
    cfg{ lid, 1 }.initB = 0.1;
    cfg{ lid, 1 }.filtersLearningRate = 1;
    cfg{ lid, 1 }.biasesLearningRate = 2;
    cfg{ lid, 1 }.filtersWeightDecay = 1;
    cfg{ lid, 1 }.biasesWeightDecay = 0;
    % Out dim: 27 * 27 * 256.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'relu';
    % Out dim: 27 * 27 * 256.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'pool';
    cfg{ lid, 1 }.method = 'max';
    cfg{ lid, 1 }.windowSize = 3;
    cfg{ lid, 1 }.stride = 2;
    cfg{ lid, 1 }.pad = 0;
    % Out dim: 13 * 13 * 256.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'normalize';
    cfg{ lid, 1 }.localSize = 5;
    cfg{ lid, 1 }.alpha = 0.0001;
    cfg{ lid, 1 }.beta = 0.75;
    % Out dim: 13 * 13 * 256.
    

    % ________________________
    % 3RD CONVOLUTIONAL LAYER.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'conv';
    cfg{ lid, 1 }.filterSize = 3;
    cfg{ lid, 1 }.filterDepth = 256;
    cfg{ lid, 1 }.numFilter = 384;
    cfg{ lid, 1 }.stride = 1;
    cfg{ lid, 1 }.pad = 1;
    cfg{ lid, 1 }.initWScal = 1;
    cfg{ lid, 1 }.initB = 0.1;
    cfg{ lid, 1 }.filtersLearningRate = 1;
    cfg{ lid, 1 }.biasesLearningRate = 2;
    cfg{ lid, 1 }.filtersWeightDecay = 1;
    cfg{ lid, 1 }.biasesWeightDecay = 0;
    % Out dim: 13 * 13 * 384.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'relu';
    % Out dim: 13 * 13 * 384.
    

    % ________________________
    % 4TH CONVOLUTIONAL LAYER.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'conv';
    cfg{ lid, 1 }.filterSize = 3;
    cfg{ lid, 1 }.filterDepth = 192;              % Shouldn't it be 384?
    cfg{ lid, 1 }.numFilter = 384;
    cfg{ lid, 1 }.stride = 1;
    cfg{ lid, 1 }.pad = 1;
    cfg{ lid, 1 }.initWScal = 1;
    cfg{ lid, 1 }.initB = 0.1;
    cfg{ lid, 1 }.filtersLearningRate = 1;
    cfg{ lid, 1 }.biasesLearningRate = 2;
    cfg{ lid, 1 }.filtersWeightDecay = 1;
    cfg{ lid, 1 }.biasesWeightDecay = 0;
    % Out dim: 13 * 13 * 384.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'relu';
    % Out dim: 13 * 13 * 384.
    

    % ________________________
    % 5TH CONVOLUTIONAL LAYER.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'conv';
    cfg{ lid, 1 }.filterSize = 3;
    cfg{ lid, 1 }.filterDepth = 192;              % Shouldn't it be 384?
    cfg{ lid, 1 }.numFilter = 256;
    cfg{ lid, 1 }.stride = 1;
    cfg{ lid, 1 }.pad = 1;
    cfg{ lid, 1 }.initWScal = 1;
    cfg{ lid, 1 }.initB = 0.1;
    cfg{ lid, 1 }.filtersLearningRate = 1;
    cfg{ lid, 1 }.biasesLearningRate = 2;
    cfg{ lid, 1 }.filtersWeightDecay = 1;
    cfg{ lid, 1 }.biasesWeightDecay = 0;
    % Out dim: 13 * 13 * 256.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'relu';
    % Out dim: 13 * 13 * 256.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'pool';
    cfg{ lid, 1 }.method = 'max';
    cfg{ lid, 1 }.windowSize = 3;
    cfg{ lid, 1 }.stride = 2;
    cfg{ lid, 1 }.pad = 0;
    % Out dim: 6 * 6 * 256.
    

    % ____________________________________
    % 1ST (APPROX.) FULLY-CONNECTED LAYER.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'conv';
    cfg{ lid, 1 }.filterSize = 6;
    cfg{ lid, 1 }.filterDepth = 256;
    cfg{ lid, 1 }.numFilter = 4096;
    cfg{ lid, 1 }.stride = 1;
    cfg{ lid, 1 }.pad = 0;
    cfg{ lid, 1 }.initWScal = 1;
    cfg{ lid, 1 }.initB = 0.1;
    cfg{ lid, 1 }.filtersLearningRate = 1;
    cfg{ lid, 1 }.biasesLearningRate = 2;
    cfg{ lid, 1 }.filtersWeightDecay = 1;
    cfg{ lid, 1 }.biasesWeightDecay = 0;
    % Out dim: 1 * 1 * 4096.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'relu';
    % Out dim: 1 * 1 * 4096.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'dropout';
    cfg{ lid, 1 }.rate = 0.5;
    % Out dim: 1 * 1 * 4096.
    

    % __________________________
    % 2ND FULLY-CONNECTED LAYER.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'conv';
    cfg{ lid, 1 }.filterSize = 1;
    cfg{ lid, 1 }.filterDepth = 4096;
    cfg{ lid, 1 }.numFilter = 4096;
    cfg{ lid, 1 }.stride = 1;
    cfg{ lid, 1 }.pad = 0;
    cfg{ lid, 1 }.initWScal = 1;
    cfg{ lid, 1 }.initB = 0.1;
    cfg{ lid, 1 }.filtersLearningRate = 1;
    cfg{ lid, 1 }.biasesLearningRate = 2;
    cfg{ lid, 1 }.filtersWeightDecay = 1;
    cfg{ lid, 1 }.biasesWeightDecay = 0;
    % Out dim: 1 * 1 * 4096.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'relu';
    % Out dim: 1 * 1 * 4096.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'dropout';
    cfg{ lid, 1 }.rate = 0.5;
    % Out dim: 1 * 1 * 4096.
    

    % __________________________
    % 3RD FULLY-CONNECTED LAYER.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'conv';
    cfg{ lid, 1 }.filterSize = 1;
    cfg{ lid, 1 }.filterDepth = 4096;
    cfg{ lid, 1 }.numFilter = 1000;
    cfg{ lid, 1 }.stride = 1;
    cfg{ lid, 1 }.pad = 0;
    cfg{ lid, 1 }.initWScal = 1;
    cfg{ lid, 1 }.initB = 0;
    cfg{ lid, 1 }.filtersLearningRate = 1;
    cfg{ lid, 1 }.biasesLearningRate = 2;
    cfg{ lid, 1 }.filtersWeightDecay = 1;
    cfg{ lid, 1 }.biasesWeightDecay = 0;
    % Out dim: 1 * 1 * 1000.

    
    % ___________
    % FINAL LOSS.
    lid = lid + 1;
    cfg{ lid, 1 }.type = 'softmaxloss';
    % Out dim: 1 * 1 * 1.
end