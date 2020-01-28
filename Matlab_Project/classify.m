%% TITLE ****************************************************************
% *                                                                      *
% *              		 521289S Machine Learning 					     *
% *                     Programming Assignment 2018                      *
% *                                                                      *
% *   Author 1: Ossi Tapio 1779165
% *                                                                      *
% *   NOTE: The file name for this file MUST BE 'classify.m'!            *
% *         Everything should be included in this single file.           *
% *                                                                      *
% ************************************************************************

%% NOTE ******************************************************************
% *                                                                      *
% *       DO NOT DEFINE ANY GLOBAL VARIABLES (outside functions)!        *
% *                                                                      *
% *       Your task is to complete the PUBLIC INTERFACE below without    *
% *       modifying the definitions. You can define and implement any    *
% *       functions you like in the PRIVATE INTERFACE at the end of      *
% *       this file as you wish.                                         *
% *                                                                      *
% ************************************************************************

%% HINT ******************************************************************
% *                                                                      *
% *       If you enable cell folding for the m-file editor, you can      *
% *       easily hide these comments from view for less clutter!         *
% *                                                                      *
% *       [ File -> Preferences -> Code folding -> Cells = enable.       *
% *         Then use -/+ signs on the left margin for hiding/showing. ]  *
% *                                                                      *
% ************************************************************************

%% This is the main function of this m-file
%  You can use this e.g. for unit testing.
%
% INPUT:  none (change if you wish)
%
% OUTPUT: none (change if you wish)
%%
function classify_exp()
tic

load trainingdata.mat %ladataan data. Nyt class_trainingData kertoo luokat, trainingData on itse data.
%load ex6_data

trainingData = trainingData; %
class_trainingData = class_trainingData; %

%parameters.trainingData = trainingData;
%parameters.training_classes = class_trainingData;

parameters = trainClassifier(trainingData, class_trainingData);
parameters
results = evaluateClassifier( trainingData, parameters );

toc
end


%% PUBLIC INTERFACE ******************************************************
% *                                                                      *
% *   Below are the functions that define the classifier training and    *
% *   evaluation. Your task is to complete these!                        *
% *                                                                      *
% *   NOTE: You MUST NOT change the function definitions that describe   *
% *         the input and output variables, and the names of the         *
% *         functions! Otherwise, the automatic ranking system cannot    *
% *         evaluate your algorithm!                                     *
% *                                                                      *
% ************************************************************************


%% This function gives the nick name that is shown in the ranking list
% at the course web page. Use 1-15 characters (a-z, A-Z, 0-9 or _).
%
% Check the rankings page beforehand to guarantee an unique nickname:
% http://www.ee.oulu.fi/research/tklab/courses/521289S/progex/rankings.html
% 
% INPUT:  none
%
% OUTPUT: Please change this to be a unique name and do not alter it 
% if resubmitting a new version to the ranking system for re-evaluation!
%%
function nick = getNickName()
    nick = 'MATLABIUS';   % CHANGE THIS!
end


%% This is the training interface for the classifier you are constructing.
%  All the learning takes place here.
%
% INPUT:  
%
%   samples:
%            A N-by-M data matrix. The rows represent samples and 
%            the columns features. N is the number of samples.
%            M is the number of features.
%
%            This could be e.g. the training data matrix given on the 
%            course web page or the validation data set that has been 
%            withheld for the validation on the server side.
%
%            Note: The value for N can vary! Do not hard-code it!
%
%   classes:
%
%            A N-by-1 vector of correct classes. Each row gives the correct
%            class for the corresponding data in the samples matrix.
%
% OUTPUT: 
%
%   parameters:
%            Any tyyppi of data structure supported by MATLAB. You decide!
%            You should use this to store the results of the training.
%
%            This set of parameters is given to the classifying function
%            that can operate on a completely different set of data.
%
%            For example, a classifier based on discriminant functions
%            could store here the weight vectors/matrices that define 
%            the functions. A kNN-classifier would store all the training  
%            data samples, their classification, and the value chosen 
%            for the k-parameter.
%            
%            Especially, structure arrays (keyword: struct) are useful for 
%            storing multiple parameters of different tyyppi in a single 
%            struct. Cell arrays could also be useful.
%            See MATLAB help for details on these.
%%
function parameters = trainClassifier( samples, classes )

N = size(samples,1); %total number of samples
num_features = size(samples,2); % how many features
k = (round(sqrt(N)/2)-1/2)*2; % how many points does knn want
mv_parzen_kerroin = 0.55; %arbitary coefficient used in parzen windows. Determined numerically beforehand.
init_feat = 1;%ceil(num_features/2); %number of initial features in SFFS-knn/parzen feature extraction
parameters.mean_samples = mean(samples);
centeredData = samples - repmat(parameters.mean_samples, N, 1); % centering the data
[V,D] = eig(cov(centeredData)); % using eigenvalue decomposition (S = VDV^-1) gain V and D matrises
W = sqrt(inv(D)) * V' ; % If VDV^T is the eigenvalue decomposition of the covariance matrix S then we get whitening matrix W
z=W* centeredData'; % apply whitening matrix to the centered data
samples = z'; %flip the dimensions so that the whitened data is in same shape as the original data

%%%%%%%%%%%%%%%% HOLDOUT VEKTORIT **************************

% saving parameters for later use
parameters.whitening_matrix = W;
parameters.N = N; 
parameters.trainingData = samples;
parameters.training_classes = classes;
parameters.k = k;
parameters.mv_parzen_kerroin = mv_parzen_kerroin;
parameters.init_feat = init_feat;

parameters.parastulos = 0.0; %initialize best result 
parameters.parasvektori = zeros(1,num_features); %initialize best vector

selection = randperm(N); % creates a random order for datapoints
training_data = samples(selection(1:floor(2*N/3)), :); %take randomly 2/3 of data as training data
validation_data = samples(selection((floor(2*N/3)+1):N), :); % rest of the data is reserved for testing
training_class = classes(selection(1:floor(2*N/3)), 1); 
validation_class = classes(selection((floor(2*N/3)+1):N), 1);


%disp('SFS knn')
[best_fvector_sfs, best_result_sfs] = SFS(training_data, training_class,k,mv_parzen_kerroin,1,num_features);
%disp('SFS parzen')
[best_fvector_sfs_parzen, best_result_sfs_parzen] = SFS(training_data, training_class,k,mv_parzen_kerroin,2,num_features);
%disp('SFFS knn')
[best_fset_knn, best_result_sffs_knn] = SFFS(training_data, training_class, k, mv_parzen_kerroin, 1,num_features);
%disp('SFFS parzen')
[best_fset_parzen, best_result_sffs_parzen] = SFFS(training_data, training_class, k, mv_parzen_kerroin, 2, num_features);

valid_res_SFS_knn = knnclass(validation_data, training_data, best_fvector_sfs', training_class, k); %use the feature vector to make predictions
correct = sum(valid_res_SFS_knn == validation_class); % amount of correct samples
validation_result_SFS_knn = correct/length(validation_class); % total accuracy of vector of this method

valid_res_SFS_parzen = parzenclass(validation_data, training_data, best_fvector_sfs_parzen', training_class, k, mv_parzen_kerroin); %use the feature vector to make predictions
correct = sum(valid_res_SFS_parzen == validation_class); % amount of correct samples
validation_result_SFS_parzen = correct/length(validation_class);% total accuracy of vector of this method

valid_res_SFFS_knn = knnclass(validation_data, training_data, best_fset_knn, training_class, k); %use the feature vector to make predictions
correct = sum(valid_res_SFFS_knn == validation_class);% amount of correct samples
validation_result_SFFS_knn = correct/length(validation_class);% total accuracy of vector of this method

valid_res_SFFS_parzen = parzenclass(validation_data, training_data, best_fset_parzen, training_class, k, mv_parzen_kerroin); %use the feature vector to make predictions
correct = sum(valid_res_SFFS_parzen == validation_class); % amount of correct samples
validation_result_SFFS_parzen = correct/length(validation_class);% total accuracy of vector of this method

%pick the best result and make the corresponding feature vector as the
%feature vector

if validation_result_SFS_knn > parameters.parastulos
    parameters.parastulos = validation_result_SFS_knn;
    parameters.parasmetodi = 1;
    parameters.parasvektori = best_fvector_sfs;
end

if validation_result_SFS_parzen > parameters.parastulos
    parameters.parastulos = validation_result_SFS_parzen;
    parameters.parasmetodi = 2;
    parameters.parasvektori = best_fvector_sfs_parzen;
end

if validation_result_SFFS_knn > parameters.parastulos
    parameters.parastulos = validation_result_SFFS_knn;
    parameters.parasmetodi = 1;
    parameters.parasvektori = best_fset_knn;
end

if validation_result_SFFS_parzen > parameters.parastulos
    parameters.parastulos = validation_result_SFFS_parzen;
    parameters.parasmetodi = 2;
    parameters.parasvektori = best_fset_parzen;
end


summa_B1 = 0.0; %initiate sums that will be used later
summa_B2 = 0.0;
summa_B3 = 0.0;
summa_B4 = 0.0;

pituus = 5; % the N in the N-fold cross validation

vektorilista_B1 = zeros(pituus, num_features); % vectorlists that are needed for the results
vektorilista_B2 = zeros(pituus, num_features);
vektorilista_B3 = zeros(pituus, num_features);
vektorilista_B4 = zeros(pituus, num_features);

for i=1:pituus
  A(i) = floor(i*N/pituus); % gives limits for N-folds bins
end

for j=1:pituus %N-fold cross-validation begins

   if j == 1
        validation_data  = samples( selection( 1 : A(j) ), :); % 1st bin is used for validation
        training_data    = samples( selection( A(j) + 1: N), :); % rest of the bins are used for training
        
        validation_class = classes( selection( 1 : A(j) ), :); %classes as above
        training_class   = classes( selection( A(j) + 1: N), :);
   elseif j == pituus
        validation_data  = samples( selection( A(j-1)+ 1: A(j) ), :); % 10th bin is used for validation
        training_data    = samples( selection( 1: A(j-1) ), :); % rest are used for training
        
        validation_class = classes( selection( A(j-1)+1: A(j)), :); %classes as above
        training_class   = classes( selection( 1: A(j-1) ), :);
   else
        training_data_A = samples(selection(1 : A(j))   ,:); % 2nd-10th bin is used for validation, rest for training
        validation_data = samples(selection(A(j) + 1: A(j + 1)),:);
        training_data_B = samples(selection(A(j + 1) + 1: N),:);
    
        training_data = [training_data_A ; training_data_B]; %unify two traninig data arrays
        
        
        training_class_A = classes(selection(1 : A(j))   ,:); %classes as above
        validation_class = classes(selection(A(j) + 1: A(j + 1)),:);
        training_class_B = classes(selection(A(j + 1) + 1: N),:);

        training_class = [training_class_A ; training_class_B];%unify two traninig data arrays
        
   end

%disp('SFS knn n-fold')
[best_fvector_sfs, best_result_sfs] = SFS(training_data, training_class,k,mv_parzen_kerroin,1,num_features);
%disp('SFS parzen n-fold')
[best_fvector_sfs_parzen, best_result_sfs_parzen] = SFS(training_data, training_class,k,mv_parzen_kerroin,2,num_features);
%disp('SFFS knn n-fold')
[best_fset_knn, best_result_sffs_knn] = SFFS(training_data, training_class, k, mv_parzen_kerroin, 1,num_features);
%disp('SFFS parzen n-fold')
[best_fset_parzen, best_result_sffs_parzen] = SFFS(training_data, training_class, k, mv_parzen_kerroin, 2, num_features);


%%%% Here we validate N-fold results
%test each vector as predictor and compare the results to actual data

valid_res_SFS_knn = knnclass(validation_data, training_data, best_fvector_sfs', training_class, k); 
correct = sum(valid_res_SFS_knn == validation_class); 
validation_result_SFS_knn = correct/length(validation_class);

valid_res_SFS_parzen = parzenclass(validation_data, training_data, best_fvector_sfs_parzen', training_class, k, mv_parzen_kerroin); 
correct = sum(valid_res_SFS_parzen == validation_class);
validation_result_SFS_parzen = correct/length(validation_class);

valid_res_SFFS_knn = knnclass(validation_data, training_data, best_fset_knn, training_class, k); 
correct = sum(valid_res_SFFS_knn == validation_class);
validation_result_SFFS_knn = correct/length(validation_class);

valid_res_SFFS_parzen = parzenclass(validation_data, training_data, best_fset_parzen, training_class, k, mv_parzen_kerroin); 
correct = sum(valid_res_SFFS_parzen == validation_class); 
validation_result_SFFS_parzen = correct/length(validation_class);

%%%% N-FOLDIN TULOSKÄSITTELY %%%%

summa_B1 = summa_B1 + validation_result_SFS_knn; %sum the results
vektorilista_B1(j,:) = best_fvector_sfs; %add vector to list of vectors

summa_B2 = summa_B2 + validation_result_SFS_parzen;
vektorilista_B2(j,:) = best_fvector_sfs_parzen;

summa_B3 = summa_B3 + validation_result_SFFS_knn;
vektorilista_B3(j,:) = best_fset_knn;

summa_B4 = summa_B4 + validation_result_SFFS_parzen;
vektorilista_B4(j,:) = best_fset_parzen;


end %N-fold cross-validation ends
tulos_B1 = summa_B1./pituus; %divide result sums with their amount in order to get averages
tulos_B2 = summa_B2./pituus;
tulos_B3 = summa_B3./pituus;
tulos_B4 = summa_B4./pituus;

vektori_B1 = mode(vektorilista_B1);
vektori_B2 = mode(vektorilista_B2);
vektori_B3 = mode(vektorilista_B3);
vektori_B4 = mode(vektorilista_B4);

%if any of the new results are better, replace the old one and the feature
%vector as well with averaged vector

if tulos_B1>parameters.parastulos 
    parameters.parastulos = tulos_B1; 
    parameters.parasmetodi = 1;
    parameters.parasvektori = vektori_B1;
end    

if tulos_B2>parameters.parastulos
    parameters.parastulos = tulos_B2;
    parameters.parasmetodi = 2;
    parameters.parasvektori = vektori_B2;
end

if tulos_B3>parameters.parastulos
    parameters.parastulos = tulos_B3;
    parameters.parasmetodi = 1;
    parameters.parasvektori = vektori_B3;
end
if tulos_B4>parameters.parastulos
    parameters.parastulos = tulos_B4;
    parameters.parasmetodi = 2;
    parameters.parasvektori = vektori_B4;
end


end


%% This is the evaluation interface of your classifier.
%  This function is used to perform the actual classification of a set of
%  samples given a fixed set of parameters defining the classifier.
%
% INPUT:   
%   samples:
%            A N-by-M data matrix. The rows represent samples and 
%            the columns features. N is the number of samples.
%            M is the number of features.
%
%            Note that N could be different from what it was in the
%            previous training function!
%
%   parameters:
%            Any tyyppi of data structure supported by MATLAB.
%
%            This is the output of the trainClassifier function you have
%            implemented above.
%
% OUTPUT: 
%   results:
%            The results of the classification as a N-by-1 vector of
%            estimated classes.
%
%            The data tyyppi and value range must correspond to the classes
%            vector in the previous function.
%%
function results = evaluateClassifier( samples, parameters )

N = size(samples,1); %total number of samples

%centeredData = samples - repmat(mean(samples), N, 1); % centering the data
centeredData = samples - repmat(parameters.mean_samples, N, 1); % centering the data
W = parameters.whitening_matrix;%sqrt(inv(D)) * V' ; % If VDV^T is the eigenvalue decomposition of the covariance matrix S then we get whitening matrix W
z=W* centeredData'; % apply whitening matrix to the centered data
samples = z'; %flip the dimensions so that the whitened data is in same shape as the original data

k = parameters.k; % parameter k
best_fvector = parameters.parasvektori; %the best vector
mv_parzen_kerroin = parameters.mv_parzen_kerroin; % arbitary parzen coefficient. Determined numerically
training_data = parameters.trainingData; %the data used to train classifier
training_class = parameters.training_classes; % classes of the data that was used to train classifier

if parameters.parasmetodi == 1 % if the best method was knn then use this classifier
    results = knnclass(samples, training_data, best_fvector, training_class, k); % valid_res equals predicted class for validation_data
else % if the best method was parzen window then use this classifier
    results = parzenclass(samples, training_data, best_fvector, training_class, k, mv_parzen_kerroin);
end

 correct = sum(results == training_class); % amount of correct samples
 validation_result = correct/length(training_class) % total accuracy of vector of this method

end


%% PRIVATE INTERFACE *****************************************************
% *                                                                      *
% *   User defined functions that are needed e.g. for training and       *
% *   evaluating the classifier above.                                   *
% *                                                                      *
% *   Please note that these are subfunctions that are visible only to   *
% *   the other functions in this file. These are defined using the      *
% *   'function' keyword after the body of the preceding functions or    *
% *   subfunctions. Subfunctions are not visible outside the file where  *
% *   they are defined.                                                  *
% *                                                                      *
% *   To avoid calling MATLAB toolbox functions that are not available   *
% *   on the server side, implement those here.                          *
% *                                                                      *
% ************************************************************************



function [predictedLabels] = knnclass(dat1, dat2, fvec, classes, k)

p1 = pdist2( dat1(:,logical(fvec)), dat2(:,logical(fvec)) ); %compare euclidean distances of two data sets
[D, I] = sort(p1', 1); %sort points by distances
I = I(1:k+1, :); % take into account only nearest k points
labels = classes( : )'; %initialize labels

    if k == 1 % if k = 1
        predictedLabels = labels( I(2, : ) )';
    else % if k is more than 1
        predictedLabels = mode( labels( I( 1+(1:k), : ) ), 1)'; % calculate the most common class of nearest k points
    end

end

function [predictedLabels_parzen] = parzenclass(valid_dat1, train_dat2, fvec, train_classes, k, mv_parzen_kerroin)

        labels = train_classes( : )'; %initialize labels
        D_parzen = pdist2( train_dat2(:,logical(fvec)), valid_dat1(:,logical(fvec)),'Chebychev' ); %the chebychev-distances between points
        [D_parzen, I_parzen] = sort(D_parzen, 1); %rearrange points according to distance
   
        predictedLabels_parzen = zeros(size(D_parzen,2),1); %initialize prediction table
        parzen_h = (nthroot(1/sqrt(length(valid_dat1(:,1))),length(valid_dat1(1,:))))*mv_parzen_kerroin; %calculate parzen window width h
        D_parzen = D_parzen./parzen_h;
        
        for n = 1:length(D_parzen(1,:)) %go through all the data points
            lkm_parzen = sum(D_parzen(:,n) < 0.5); % how many points are in the chebychev-distance
            list_parzen = I_parzen(2:lkm_parzen,n); % measure which points are within desired distance from the measuring point
            classlist_parzen = labels(list_parzen); % resolve the class of detected points
            predictedLabels_parzen(n,1) = mode(classlist_parzen); %calculate which class is most represented in the measured distance

        end

end

function [best, feature] = findbest_combo(data, data_c, fvector, direction, k, mv_parzen_kerroin, tyyppi)

num_samples = length(data);% number of total data points
best = 0; %initialze best result
feature = 0;% which vector feature is being analyzed
labels = data_c( : )'; % initialize labels 
parzen_h = (nthroot(1/sqrt(length(data(:,1))),length(data(1,:))))*mv_parzen_kerroin; % this is the size of the parzen window
for in = 1:length(fvector)

    if (direction == 0 && fvector(in) == 0 ||direction == 1 && fvector(in) == 1 )
%        tic
        if direction == 0 %include the vec tor element corresponding to 'in' 
            fvector(in) = 1; % here we set the vector element corresponding to 'in' as one
                             %'fvector' is then used as a mask to choose the coulmns from the data matrix which are ones in 'fvector'
        else %exclude the vector element corresponding to 'in' 
            fvector(in) = 0;% here we set the vector element corresponding to 'in' as zero
        end
        
        if tyyppi == 1 % if tyyppi is 1 use knn classifier
            predictedLabels = find_knn(data, fvector, labels, k);
        end
        
        if tyyppi == 2 % if tyyppi is 2 use parzen classifier
            predictedLabels = find_parzen(data, data_c, fvector, labels, parzen_h);          
        end
            
        correct = sum(predictedLabels == data_c); % resolve which predictions were correct and sum them
        result = correct/num_samples; % compare total correct predictions to total number of data points

        if(result > best) % if we have better result than before, save them
            best = result; % the best result we have got so far
            feature = in; % the particular feature that caused the best result
        end

        if direction == 0 %change the feature vector in its original shape
            fvector(in) = 0; 
        else
            fvector(in) = 1;
        end

    end

end

end

function [predictedLabels] = find_knn(data, fvector, labels, k)
    D = pdist2( data(:,logical(fvector)), data(:,logical(fvector)) );

    [D, I] = sort(D, 1); %rearrange points according to distance
    I = I(1:k+1, :); %take only nearest k points into account
           
    predictedLabels = mode( labels( I( 1+(1:k), : ) ), 1)'; % count which class of points is most near which point

end

function [predictedLabels] = find_parzen(data, data_c, fvector, labels, parzen_h)

    D_parzen = pdist2( data(:,logical(fvector)),data(:,logical(fvector)) ,'Chebychev' ); %the chebychev-distances between points
    [D_parzen, I_parzen] = sort(D_parzen, 1); %rearrange points according to distance
    predictedLabels = zeros(length(data_c),1); %initialize prediction table
    D_parzen = D_parzen./parzen_h;
    for n = 1:length(data_c) %go through all the data points
        lkm_parzen = sum(D_parzen(:,n) < 0.5); % how many points are in the chebychev-distance
        list_parzen = I_parzen(2:lkm_parzen,n); % measure which points are within desired distance from the measuring point
        classlist_parzen = labels(list_parzen); % resolve the class of detected points
        predictedLabels(n,1) = mode(classlist_parzen); %calculate which class is most represented in the measured distance
    end

end

function [best_fvector, best_result] = SFS(training_data, training_class,k,mv_parzen_kerroin,tyyppi, num_features)
%disp('SFS function')
fvector = zeros(num_features,1); %initialize vectors for SBS and SFS feature selection with knn
best_result = 0; % best result

for in = 1:num_features %go through desired amount of features
    
    [best_result_add, best_feature_add] = findbest_combo(training_data, training_class, fvector, 0, k, mv_parzen_kerroin, tyyppi);
    fvector(best_feature_add) = 1; %update feature vector
    
    if(best_result < best_result_add) % if new result is better, save the new vector
        best_result = best_result_add;
        best_fvector = fvector;
    end
    
end
best_fvector = best_fvector'; % change the orientation of the vector
end


function [best_fset, best_result] = SFFS(training_data, training_class, k, mv_parzen_kerroin, tyyppi, num_features)

fvector_sffs = zeros(1,num_features);%zeros(1,size(training_data, 2)); % initialize feature vectors
n_features = 1;%init_feat; % how many initial features are investigated
best_result = 0; % best results of vectors
res_vector = zeros(1,num_features); % tulosvektorien alustus
search_direction = 0; % initial search direction for better result , 0 forward, 1 backward

while(n_features <= num_features) % go through how many features are wanted in total
                                                           
    [best_result_add, best_feature_add] = findbest_combo(training_data, training_class, fvector_sffs, search_direction,k, mv_parzen_kerroin,tyyppi); %try find better feature vector (forwards)
    
    fvector_sffs(best_feature_add) = 1; %update feature vector

    if(best_result < best_result_add) %if result is better, save new feature 
       best_result = best_result_add;
       best_fset = fvector_sffs;
    end

    if(best_result_add > res_vector(n_features)) %if the new feature vector has same number of features as the old one but gives better results, save the new one
        res_vector(n_features) = best_result_add;
    end

 	search_direction = 1; %switch search direction. trying to exclude a feature.

   	while search_direction

            if(n_features > 2) % if there is more than one feature. proceed
            	
                [best_result_rem, best_feature_rem] = findbest_combo(training_data, training_class, fvector_sffs, search_direction,k, mv_parzen_kerroin,tyyppi); 
            	% try to improve feature vector by removing feature
            	
            	if(best_result_rem > res_vector(n_features - 1)) % compare reduced feature vector to the previous step
                    fvector_sffs(best_feature_rem) = 0;
                    n_features = n_features - 1;
                    if(best_result < best_result_rem) % if better reduced feature vector is better than previous step, reduce one feature
                        best_result = best_result_rem;
                        best_fset = fvector_sffs;
                    end
                    
                    res_vector(n_features) = best_result_rem; %update results
                
                else
                    search_direction = 0; % if improvement could not be made by removing features, change direction
            	end
            
            else
                search_direction = 0; %if not enough features, change search direction
            end
        
     end
      n_features = n_features + 1; %add the number of features investigated in the next step
    
end

end
