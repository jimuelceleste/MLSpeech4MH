
def nested_cv(input_dir):
	features, labels = get_data(input_dir)

	pipeline = Pipeline([
		('feature_scaler', StandardScaler()), 
		('feature_selector', SelectKBest()),
		('model', RandomForestClassifier())
	])

	pipeline_parameter_space = {
		'selector__k': [10, 20, 50],
		'model__n_estimators': [100, 300, 500, 700],
		'model__max_features': [5, 15, 25, 35, 45, 55],
		'model__min_samples_leaf': [1, 2, 3, 4]
	}

	# Model Selection Folds
	model_selection_folds = generate_subject_wise_folds(features, labels, k=5)

	for train, validate in model_selection_folds:
 
		# Hyperparameter Optimization Folds
		splitter = StratifiedGroupKFold(n_splits=5, random_state=52, shuffle=True)
		features_train, labels_train, subject_id_train = train
        inner_cv_folds = splitter.split(
        	X=features_train, 
        	y=labels_train, 
        	groups=subjects_id_train
        )

        # Run Bayesian Search
        optimizer = BayesSearchCV(
        	pipeline, 
        	search_spaces=pipeline_parameter_space, 
        	cv=inner_cv_folds, 
        	scoring='f1_macro',
        	verbose=True,
        	n_jobs=-1
        )
        optimizer.fit(features_train, labels_train)

        # Run Grid Search Optimization
        optimizer = GridSearchCV(
        	estimator=pipeline,
        	param_grid=pipeline_parameter_space,
        	cv=inner_cv_folds,
        	scoring='f1_macro',
        	verbose=True,
        	n_jobs=-1
        )
        optimizer.fit(features_train, labels_train)

        # model fitting 
        best_clf = optimizer.best_estimator_.fit(features_train, labels_train)
        best_hyp = optimizer.best_params_

        # validation
        y_pred = best_clf.predict(X_val)
        macro_f1 = f1_score(y_val, y_pred, average='macro')
        macro_precision = precision_score(y_val, y_pred, average='macro')
        macro_recall = recall_score(y_val, y_pred, average='macro')
        print('macro_f1', macro_f1, 'macro_precision', macro_precision, 'macro_recall', macro_recall)
        
        # artifact preservation
        output_dir_fold = os.path.join(output_dir_experiment, f'fold_{fold}')
        os.makedirs(output_dir_fold, exist_ok=True)

        # Predictions
        preds = pd.DataFrame.from_dict({'label': samples_val, 'true': y_val, 'pred': y_pred})
        preds.to_csv(os.path.join(output_dir_fold, f'predictions.csv'))

        # Performance
        perf = {'macro_f1': macro_f1, 'macro_precision': macro_precision, 'macro_recall': macro_recall}
        perf = pd.DataFrame(perf, index=[fold])
        perf.to_csv(os.path.join(output_dir_fold, f'performance.csv'))
        performance.append(perf)

        # Hyperparameters 
        hyp = pd.DataFrame(best_hyp, index=[fold])
        hyp.to_csv(os.path.join(output_dir_fold, f'hyperparameters.csv'))
