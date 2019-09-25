clc;
clear all;
addpath('../src');

dataSet = ["Cryotherapy"];

for data_num=1:length(dataSet)
    dataset_name = char(dataSet(data_num));
    disp(dataset_name);
    runlayers=[1 3]; nclust=10;

    %Defining number of folds for k-fold cross validation
    kf = 5;

    % Range of all parameter            
    Range_gamma = 0;
    Range_rr = 1;
    Range_C = power(2,-5:5);
    Range_Cl = power(2,-5:5);

    if ~exist(['Results/benchmarkSmall/' dataset_name])
       mkdir (['Results/benchmarkSmall/' dataset_name])
    end

    %%% load dataset
    load(['../src/data/benchmarkSmall/' dataset_name]);
    train_pos_data_all = train_pos_data_all';

    %%%Train-test data Normalization %%%
    [ntrain_data_all norm_valt] = norm_denorm(train_pos_data_all', 2, 1);
    ntest_data = norm_denorm(test_data', 2, 0, norm_valt);
    train_pos_data_all = ntrain_data_all';
    test_data = ntest_data'; 
    clear ntrain_data_all ntest_data;
    %%% End of Normalization %%%

    for fld = 1:kf
        train_data = folds_train(fld).train_pos';
        val_data = folds_train(fld).val';
        val_labels = folds_train(fld).val_labels;

        %%%Train-validation data Normalization %%%
        [ntrain_data norm_val] = norm_denorm(train_data', 2, 1);
        nval_data = norm_denorm(val_data', 2, 0, norm_val);
        clear train_data val_data;
        train_data = ntrain_data';
        val_data = nval_data'; 
        clear ntrain_data nval_data;
        %%% End of Normalization %%%

        for layer=1:length(runlayers)
            Laplacian = [];
            for i=1:length(Range_gamma)
                gama = Range_gamma(i);
                for j=1:length(Range_C)
                    C = Range_C(j);
                    for k=1:length(Range_Cl)
                        Cl = Range_Cl(k);

                        %%%% OCKELM and ML-OCKELM
                        [labelval_GOCKML_tmp ~] = MLOCKELM(train_data,val_data,runlayers(layer),C,Cl,gama);
                        for muIndex=1:3     %Iterates over possible fractions of rejection, namely, [1% 5% 10%]
                            labelval_GOCKML = labelval_GOCKML_tmp(:,muIndex);
                            [accuMLval(fld,layer,i,j,k,muIndex) sensMLval(fld,layer,i,j,k,muIndex) specMLval(fld,layer,i,j,k,muIndex)...
                                precMLval(fld,layer,i,j,k,muIndex) recMLval(fld,layer,i,j,k,muIndex) f11MLval(fld,layer,i,j,k,muIndex)...
                                gmMLval(fld,layer,i,j,k,muIndex)] = Evaluate(val_labels,labelval_GOCKML,1);
                        end

                        for gl=1:length(Range_rr)
                            rr = Range_rr(gl); 
                            for noofcluster=1:nclust

                                %%%% VOCKELM and DKRLVOC
                                Laplacian = [];
                                [labelval_GOCKDKRL_tmp ~] = DKRLVOC(train_data,val_data,runlayers(layer),C,Cl,gama,noofcluster,Laplacian,rr);                                                               
                                for muIndex=1:3     %Iterates over possible fractions of rejection, namely, [1% 5% 10%]
                                    labelval_GOCKDKRL = labelval_GOCKDKRL_tmp(:,muIndex);
                                    [accuDKRLval(fld,layer,i,j,k,gl,noofcluster,muIndex) sensDKRLval(fld,layer,i,j,k,gl,noofcluster,muIndex)....
                                        specDKRLval(fld,layer,i,j,k,gl,noofcluster,muIndex) precDKRLval(fld,layer,i,j,k,gl,noofcluster,muIndex)...
                                        recDKRLval(fld,layer,i,j,k,gl,noofcluster,muIndex) f11DKRLval(fld,layer,i,j,k,gl,noofcluster,muIndex)...
                                        gmDKRLval(fld,layer,i,j,k,gl,noofcluster,muIndex)] = Evaluate(val_labels,labelval_GOCKDKRL,1);
                                end
                                clear Laplacian;
                            end
                        end
                    end
                end
            end       
            disp([dataset_name ' Fold: ' num2str(fld) ' Layer: ' num2str(runlayers(layer))])
        end   
    end

    %%%% Save validation results
    save(['Results/benchmarkSmall/' dataset_name '/' dataset_name '_valNOG'], 'accuMLval','sensMLval','specMLval','precMLval','recMLval','f11MLval','gmMLval');
    save(['Results/benchmarkSmall/' dataset_name '/' dataset_name '_valMSCV'], 'accuDKRLval','sensDKRLval','specDKRLval','precDKRLval','recDKRLval','f11DKRLval','gmDKRLval'); 

    mu = [0.01 0.05 0.1];

    %Defining cell arrays to save optimum parameters
    param_ML = cell(1,2);
    param_MLSV = cell(1,2);

    for layeridx =1:2
        for frej=1:3
            for i=1:length(Range_gamma)
                for j=1:length(Range_C)
                    for k=1:length(Range_Cl)
                        avg_gmMLval(layeridx,i,j,k,frej) = mean(gmMLval(:,layeridx,i,j,k,frej));

                        for gl=1:length(Range_rr)
                            for noofcluster=1:nclust
                                avg_gmDKRLval(layeridx,i,j,k,gl,noofcluster,frej) = mean(gmDKRLval(:,layeridx,i,j,k,gl,noofcluster,frej));
                            end
                        end
                    end
                end
            end
        end

        %%%%  OCKELM & MLOCKELM
        gmMLval_1 = avg_gmMLval(layeridx,:,:,:,:);
        [max_gmMLval indmax]=max(gmMLval_1(:));
        %%% Order of name of parameter in these indexes layer,Range_gamma,Range_C,Range_Cl,frej
        [gind1 gind2 gind3 gind4 gind5]=ind2sub(size(gmMLval_1),indmax);
        param_ML{layeridx} = ["Range_gamma" "Range_C" "Range_Cl" "frej"; ...
            Range_gamma Range_C(gind3) Range_Cl(gind4) mu(gind5)];

        %%% For VOCKELM & DKRLVOC
        gmDKRLval_1 = avg_gmDKRLval(layeridx,:,:,:,:,:,:);
        [max_gmDKRLval indmax] =max(gmDKRLval_1(:));
        %%% Order of name of parameter in these indexes layer,Range_gamma,Range_C,Range_Cl,Range_rr,noofcluster
        [CVgind1 CVgind2 CVgind3 CVgind4 CVgind5 CVgind6 CVgind7]=ind2sub(size(gmDKRLval_1),indmax);
        noofcluster=1:nclust;
        param_MLSV{layeridx} = ["Range_gamma" "Range_C" "Range_Cl" "Range_rr" "noofcluster" "frej"; ...
            Range_gamma Range_C(CVgind3) Range_Cl(CVgind4) Range_rr noofcluster(CVgind6) mu(CVgind7)];
    end

    %Begin Testing phase
    for layer=1:length(runlayers)
        %%%% OCKELM and ML-OCKELM
        %Setting parameters
        tmp = param_ML{layer};
        gama = str2num(cell2mat(tmp(2,1)));
        C = str2num(cell2mat(tmp(2,2)));
        Cl = str2num(cell2mat(tmp(2,3)));
        %Calling model
        [labels_GOCKML_tmp, ~] = MLOCKELM(train_pos_data_all,test_data,runlayers(layer),C,Cl,gama); 
        labels_GOCKML = labels_GOCKML_tmp(:,mu==str2num(cell2mat(tmp(2,4))));
        [accuML(layer) sensML(layer) specML(layer) precML(layer) recML(layer) f11ML(layer) ...
            gmML(layer)] = Evaluate(test_labels,labels_GOCKML,1);
        clear tmp;

        %%%% VOCKELM and DKRLVOC
        Laplacian = [];
        %Setting parameters
        tmp = param_MLSV{layer};
        gama = str2num(cell2mat(tmp(2,1)));
        C = str2num(cell2mat(tmp(2,2)));
        Cl = str2num(cell2mat(tmp(2,3)));
        rr = str2num(cell2mat(tmp(2,4)));
        noofcluster = str2num(cell2mat(tmp(2,5)));
        %Calling model
        [labels_GOCKDKRL_tmp, ~] = DKRLVOC(train_pos_data_all,test_data,runlayers(layer),C,Cl,gama,noofcluster,Laplacian,rr);
        labels_GOCKDKRL = labels_GOCKDKRL_tmp(:,mu==str2num(cell2mat(tmp(2,6))));
        [accuDKRL(layer) sensDKRL(layer) specDKRL(layer) precDKRL(layer) recDKRL(layer) f11DKRL(layer)...
            gmDKRL(layer)] = Evaluate(test_labels,labels_GOCKDKRL,1);
        clear Laplacian tmp;
    end

    f11 = [round(f11ML(1)*100,2) round(f11DKRL(1)*100,2) round(f11ML(2)*100,2) round(f11DKRL(2)*100,2)];
    gm = [round(gmML(1)*100,2) round(gmDKRL(1)*100,2) round(gmML(2)*100,2) round(gmDKRL(2)*100,2)];
    accu = [round(accuML(1)*100,2) round(accuDKRL(1)*100,2) round(accuML(2)*100,2) round(accuDKRL(2)*100,2)];
    prec = [round(precML(1)*100,2) round(precDKRL(1)*100,2) round(precML(2)*100,2) round(precDKRL(2)*100,2)];
    rec = [round(recML(1)*100,2) round(recDKRL(1)*100,2) round(recML(2)*100,2) round(recDKRL(2)*100,2)];

    save(['Results/benchmarkSmall/' dataset_name '/Results.mat'],'dataset_name','f11','gm','accu','prec','rec');

    disp(' ')     
    disp(['F1 Score for OCKELM: ' num2str(f11(1))]);
    disp(['F1 Score for VOCKELM: ' num2str(f11(2))]);
    disp(['F1 Score for ML-OCKELM: ' num2str(f11(3))]);
    disp(['F1 Score for DKRLVOC: ' num2str(f11(4))]);
    
    disp(' ')  
    disp(['Optimal Parameters selected for DKRLVOC after cross-validation:']);
        tmp = param_MLSV{2};
        disp(['     Number of layers: ' num2str(runlayers(2))]);
        disp(['     Regularization parameter for 1st two layers (C): ' cell2mat(tmp(2,2))]);
        disp(['     Regularization parameter for last layer (Cf): ' cell2mat(tmp(2,3))]);
        disp(['     Graph Regularization parameter (lambda): ' cell2mat(tmp(2,4))]);
        disp(['     Number of clusters (k): ' cell2mat(tmp(2,5))]);
        disp(['     Percentage of dismissal (delta): ' cell2mat(tmp(2,6))]);
end