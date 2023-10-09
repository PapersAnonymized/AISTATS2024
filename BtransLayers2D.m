classdef BtransLayers2D
    properties
    end

    methods

        function net = BtransLayers2D()
        end


        function net = Create(net)

            layers = [
                featureInputLayer(net.m_in)
                fullyConnectedLayer(net.k_hid1,'Name','inputFeatureExt')

                bTransformerLayer(net.k_hid1, "k_hid1")
                layerNormalizationLayer('Name','fcNorm')
                fullyConnectedLayer(net.k_hid1)              
                
                fullyConnectedLayer(net.k_hid2,'Name','FeatureExt2')
                
                bTransformerLayer(net.k_hid2, "k_hid2")
                layerNormalizationLayer('Name','fcNorm2')
                fullyConnectedLayer(net.k_hid2) 

                fullyConnectedLayer(net.n_out)
                regressionLayer
            ];

            net.lGraph = layerGraph(layers);

            net.options = trainingOptions('adam', ...
                'ExecutionEnvironment','auto',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', net.mb_size, ...
                'InitialLearnRate', net.ini_rate, ...
                'MaxEpochs',net.max_epoch);

                 %'Plots', 'training-progress',...


        end

        
    end
end