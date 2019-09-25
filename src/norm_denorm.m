
function [out_val, norm_denorm_val] = norm_denorm (pos_neg, norm_type, train_norm, nd_val)

%%%%%% Normalize in the range [0 1] %%%%%%%
%%%% Input %%%%
%%% pos_neg = positive or negative i.e. any data which need to be normalized
%%% norm_type = Type of normalization:
%             1: Min-Max Normalization, Normalize in the range [0 1]
%             2: ZScore
%%% train_norm = traninig or testing data:
%   1= training;
%   0= testing;
%   2= denormalization;
%%%
%%% nd_val = it's a cell array which contains value which can be used
% to normalize or denormalize the testing or any other data. This value is obtained
% during normalization of training data as norm_denorm_val

%%%% Output %%%%
%%% out_val = normalized or denormalized data
%%% norm_denorm_val = value obtained from training data which will
%%% be utilized for normalizing or denormalizing the testing or any other data

if (nargin<3) train_norm = 1; end
if (nargin<2) train_norm = 1; norm_type = 1; end
if (nargin<1) error('pass atleast 1 arguments as a training set'); end

switch norm_type
    
    case {1}
        %%%%% Min-Max Normalization %%%%
        if (train_norm==1)
            pos_class = pos_neg;
            minimums = min(pos_class, [], 1);
            maximums = max(pos_class, [], 1);
            ranges = maximums - minimums;
            ind = find(ranges==0);
            ranges(ind) = 1; % maximums(ind) + 0.000001;
            norm_denorm_val{1} = ranges;
            norm_denorm_val{2} = minimums;
            norm_denorm_val{3} = maximums;
            out_val = (pos_class - repmat(minimums, size(pos_class, 1), 1)) ./ repmat(ranges, size(pos_class, 1), 1);
            
        elseif(train_norm==0)
            %%%%% For testing or any other data %%%%%
            posneg_class = pos_neg;
            ranges = nd_val{1};
            minimums = nd_val{2};
            out_val = (posneg_class - repmat(minimums, size(posneg_class, 1), 1)) ./ repmat(ranges, size(posneg_class, 1), 1);
            
        elseif(train_norm==2)
            %%%%% To denormalize the data %%%%%
            todenorm = pos_neg;
            ranges = nd_val{1};
            minimums = nd_val{2};
            out_val = (todenorm  .* repmat(ranges, size(todenorm, 1), 1)) + repmat(minimums, size(todenorm, 1), 1);
            
        else
            disp('Provide any appropriate poarameter value: 1= training; 0= testing; 2= denormalization');
            
        end
        
    case {2}
        %%%%% ZScore Normalization %%%%
        if (train_norm==1)
            pos_class = pos_neg;
            mean_pos = mean(pos_neg);
            std_pos = std(pos_neg);
            % if std of a feature is 0 then do not divide it
            std_pos(std_pos == 0) = 1;
            norm_denorm_val{1} = mean_pos;
            norm_denorm_val{2} = std_pos;
            out_val = (pos_class - repmat(mean_pos, size(pos_class, 1), 1)) ./ repmat(std_pos, size(pos_class, 1), 1);     
            
        elseif(train_norm==0)
            %%%%% For testing or any other data %%%%%
            posneg_class = pos_neg;
            mean_pos = nd_val{1};
            std_pos = nd_val{2};
            out_val = (posneg_class - repmat(mean_pos, size(posneg_class, 1), 1)) ./ repmat(std_pos, size(posneg_class, 1), 1);
            
        elseif(train_norm==2)
            %%%%% To denormalize the data %%%%%
            todenorm = pos_neg;
            mean_pos = nd_val{1};
            std_pos = nd_val{2};
            out_val = (todenorm  .* repmat(std_pos, size(todenorm, 1), 1)) + repmat(mean_pos, size(todenorm, 1), 1);
            
        else
            disp('Provide any appropriate poarameter value: 1= training; 0= testing; 2= denormalization');
            
        end
end
end