classdef TanhNet2Cl < TanhLayers2Cl & BaseNet2Cl

    properties

    end

    methods
        function net = TanhNet2Cl(x_off, x_in, t_in, n_sess, ini_rate, max_epoch)

            net = net@TanhLayers2Cl();
            net = net@BaseNet2Cl(x_off, x_in, t_in, n_sess, ini_rate, max_epoch);

            net.name = "tanh2cl";

        end

        
    end
end