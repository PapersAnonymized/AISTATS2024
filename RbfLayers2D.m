classdef RbfLayers2D
    properties
    end

    methods

        function net = RbfLayers2D()
        end


        function net = Create(net)

            layers = [
                featureInputLayer(net.m_in)
                GaussianRBFLayer('RBF1', net.m_in, net.k_hid1)
                GaussianRBFLayer('RBF2', net.k_hid1, net.k_hid2)
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