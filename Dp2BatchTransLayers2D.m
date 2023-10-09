classdef Dp2BatchTransLayers2D
    properties
    end

    methods

        function net = Dp2BatchTransLayers2D()
        end


        function net = Create(net)

            layers = [
                featureInputLayer(net.m_in)
                fullyConnectedLayer(net.k_hid1,'Name','inputFeatureExt')

                dpBatchTransformerLayer(net.k_hid1, "b_k_hid1")
                dpTransformerLayer(net.k_hid1, "k_hid1")
                fullyConnectedLayer(net.k_hid1) 

                fullyConnectedLayer(net.k_hid2,'Name','FeatureExt2')
                
                dpBatchTransformerLayer(net.k_hid2, "b_k_hid2")
                dpBatchTransformerLayer(net.k_hid2, "k_hid2")
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