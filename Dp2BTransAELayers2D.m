classdef Dp2BTransAELayers2D
    properties
    end

    methods

        function net = Dp2BTransAELayers2D()
        end


        function net = Create(net)

            layers = [
                featureInputLayer(net.m_in)
                fullyConnectedLayer(net.k_hid1,'Name','inputFeatureExt')

                %%dpTransformerLayer(net.k_hid1, "k_hid1")
                %dpBatchTransformerLayer(net.k_hid1, "b_k_hid1")
                %%dpTransformerLayer(net.k_hid1, "k_hid1")

                dp2BatchTransformerLayer(net.k_hid1, "b_k_hid1")

                fullyConnectedLayer(net.n_out,'Name','FeatureBottle') 

                reluLayer
                %tanhLayer
                fullyConnectedLayer(net.n_out)
                reluLayer
                %tanhLayer
                fullyConnectedLayer(2*net.n_out+1)

                %GaussianRBFLayer('RBF1', net.n_out, net.n_out)
                %GaussianRBFLayer('RBF2', net.n_out, 2*net.n_out+1)

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