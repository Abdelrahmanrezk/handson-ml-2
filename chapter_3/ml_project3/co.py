

# num_imputer = ColumnTransformer([
#     ('mean_imputer', SimpleImputer(strategy='mean'), num_attr_names[:2]),
#     ('mode_imputer', SimpleImputer(strategy='most_frequent'), num_attr_names[2:]),
# ])
# num_pipeline = Pipeline([
#     ('outliers', LeaveOrRemoveOutLiers(['SibSp', 'Parch'], False)),
# #     ('scaling', FeatureScale(num_attr_names, False)),
# #     ('imputer', num_imputer),
# ])
# cat_pipeline = Pipeline([
#     ('HotEncoding', OneHotEncoder(), cat_attr_names),
# ])
# # num_pipeline = num_pipeline.fit_transform(titanic_tr_pipe_line)

# # cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# # cat_imputer = cat_imputer.fit_transform(titanic_tr_pipe_line[cat_attr_names].values)
# # titanic_tr_pipe_line[cat_attr_names] = cat_imputer

# # cat_pipeline = cat_pipeline.fit_transform(titanic_tr_pipe_line)
# # nan_scale_outliers_pipeline = np.c_[num_pipeline, cat_pipeline]
# # print(num_pipeline.shape)
# # print(cat_pipeline.shape)
# # print(nan_scale_outliers_pipeline.shape)

# # nan_scale_outliers_pipeline = ColumnTransformer([
# #     ('num_pipeline', num_pipeline, num_attr_names), 
# # #     ('cat_pipeline', cat_pipeline, cat_attr_names),
    
# # ])


