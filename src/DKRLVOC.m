%Contains code for VOCKELM(layer=1) and DKRLVOC(layer>1)

function [labels labeltr layerinfo] = DKRLVOC(train_data,test_data,max_layer,C,Cl,gamma,noofclusters,L,lambda)
    % Embedding Minimum Sub-class Variance
    for layer =1:max_layer
        if layer==1
            [K, Ktest, ~] = kernel_rbf(train_data,test_data,gamma);
            II = eye(size(K,1));
				
            S = mscv_calc(K, ones(size(K,2),1), noofclusters);
            if ~isempty(L)
                S = S+L;
            end
            
            if nargin < 8 
                B = K + II/C;
            elseif nargin == 8 
                B = K + (S*K)/C;
            elseif nargin ==9 
                B = K + (lambda/C)*(S*K) + II/C;
            end
            
            if (layer~=max_layer)
                a = pinv(B) * train_data';
                Ot = a' * Ktest;
                Otr = a' * K;
            else
                Ot = test_data;
                Otr = train_data;            
            end
        end
        
        if layer==max_layer
            clear B a K Ktest;
            [K, Ktest, ~] = kernel_rbf(Otr,Ot,gamma);
            II = eye(size(K,1));
        
            if (max_layer==1)
                if nargin < 8
                    B = K + II/Cl;
                elseif nargin == 8 
                    B = K + (S*K)/Cl;
                elseif nargin ==9
                    B = K + (lambda/Cl)*(S*K) + II/Cl;
                end
            else
                if nargin < 8 
                    B = K + II/Cl;
                elseif nargin == 8
                    B = K + (L*K)/Cl;
                elseif nargin ==9
                    if ~isempty(L)
                        B = K + (lambda/Cl)*(L*K) + II/Cl;
                    else
                        B = K + II/Cl;
                    end
                end
            end
        
            try
                a = pinv(B) * ones(size(K,2),1);
            catch
                warning('Problem with pinv, SVD did not converge');
                a = inv(B) * ones(size(K,2),1);
            end
            clear Ot Otr;
            Ot = a' * Ktest;
            Otr = a' * K;
        elseif ((layer~=max_layer) & (layer~=1))
            clear B a K Ktest;
            [K, Ktest, ~] = kernel_rbf(Otr,Ot,gamma);
            II = eye(size(K,1));
        
            if nargin < 8
                B = K + II/C;
            elseif nargin == 8
                B = K + (L*K)/C;
            elseif nargin ==9
                if ~isempty(L)
                    B = K + (lambda/C)*(L*K) + II/C;
                else
                    B = K + II/C; 
                end
            end
            try
                a = pinv(B) * Otr';
            catch
                warning('Problem with pinv, SVD did not converge');
                a = inv(B) * Otr';
            end
            clear Ot Otr;        
            Ot = a' * Ktest;
            Otr = a' * K;
        end
    
        %%%% Just gathering the information regarding outputweight of all layers and output
        %%%% of autoencoder at each layer except last layer as last is not an
        %%%% autoencoder but calculates final output.
        if layer==max_layer
            OutputWeight_max_layers= a';
        else
            OutputWeights(:,:,layer)= a';
            layer_autoenc_trains(:,:,layer)= Otr;
            layer_autoenc_tests(:,:,layer)= Ot;
        end
    end

    %%% Just storing all information into a single variable as a structre
    layerinfo.OutputWeight_max_layer=OutputWeight_max_layers;
    if (layer~=1)
        layerinfo.OutputWeight=OutputWeights;
        layerinfo.OutputWeight_max_layer=OutputWeight_max_layers;
        layerinfo.layer_autoenc_train=layer_autoenc_trains;
        layerinfo.layer_autoenc_test=layer_autoenc_tests;
    end

    %%% One-class Classification as per single node output %%%
    %%% For training
    mu=[0.01 0.05 0.1];
    m = size(train_data,2);
    difftr = abs(ones(m,1)-Otr');
    [sout,~] = sort(difftr);
    for muIndex=1:3
        labeltr_temp = ones(m,1)*2; 
        thresh = sout(ceil(m*(1-mu(muIndex))),1);
        labeltr_temp(difftr<thresh)=1;
        labeltr(:,muIndex) = labeltr_temp;

        %%% For Testing
        labels_temp = ones(size(test_data,2),1)*2;
        diffs = abs(ones(size(test_data,2),1)-Ot');
        labels_temp(diffs<thresh)=1;
        labels(:,muIndex) = labels_temp;
    end
end