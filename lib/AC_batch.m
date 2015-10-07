
function [F, labels, entries, acc, map] = AC_batch (db_params, features_params, learning_params, summarization_params, ...
    equalization_params, classification_params, Nfolds)
saved_features = ''; % to decide if features must be recomputed
saved_db = '';
all_p = sprintf ('db_params:\n%s\nfeatures_params:\n%s\nlearning_params:\n%s\nsummarization_params:\n%s\nequalization_params:\n%s\nclassification_params:\n%s\n', ...
    evalc (['disp (db_params)']), evalc (['disp (features_params)']), evalc (['disp (learning_params)']), ...
    evalc (['disp (summarization_params)']), evalc (['disp (equalization_params)']), evalc (['disp (classification_params)']));

%% compute features
tstart = tic;
if exist('AC_features_and_labels.mat', 'file') == 2
    load('AC_features_and_labels.mat');
end

if (strcmp (saved_features, evalc (['disp (features_params)'])) ~= true) || ...
        (strcmp (saved_db, evalc (['disp (db_params)'])) ~= true)
    [F, labels, entries] = AC_features (db_params.location, features_params);
    saved_features = evalc (['disp (features_params)']);
    saved_db = evalc (['disp (db_params)']);
    save('AC_features_and_labels.mat', 'F', 'labels', 'entries', 'saved_features', 'saved_db', '-v7.3');
end

Nclass = length (unique (labels));

%% learning and transformations
[F, kernels] = AC_learning (F, learning_params);

%% summarization
[F, labels, entries] = AC_summarization (F, labels, entries, summarization_params);
save('AC_data_summarized.mat', 'F', 'labels', 'entries', '-v7.3');

%% distribution equalization (in number of samples per class)
[F, labels, entries] = AC_distribution_eq (F, labels, entries, Nclass, equalization_params);

%% classification

accv = zeros (Nfolds,1);
mapv = zeros (Nfolds,1);

parfor ifold = 1:Nfolds
    fprintf ('classification fold %d\n', ifold);
    [accv(ifold), mapv(ifold)] = AC_classification (F, labels, entries, Nclass, classification_params);       
end

acc = mean (accv); map = mean (mapv); 

telapsed = toc (tstart);
fprintf ('\n');
results = sprintf ('[final results]\nacc = %f, map = %f, (performance time: %f sec.)\n', acc, map, telapsed);
disp (results);

%% store results
AC_storage (all_p, results);

end

%% eof
