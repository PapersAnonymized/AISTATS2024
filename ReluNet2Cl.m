classdef ReluNet2Cl < ReluLayers2Cl & BaseNet2Cl

    properties

    end

    methods
        function net = ReluNet2Cl(x_off, x_in, t_in, n_sess, ini_rate, max_epoch)

            net = net@BaseNet2Cl(x_off, x_in, t_in, n_sess, ini_rate, max_epoch);
            net = net@ReluLayers2Cl();

            net.name = "relu2cl";

        end

        
    end
end