
# Product Review Analysis - FlipKart_Amazon

E-commerce is one of the booming industries & is a one-stop destination for various sellers to market & sell their products online to attract a larger market. Given a set of customer reviews of each category (camera, battery, display, value for money, performance) for a mobile that is live on an e-commerce platform like (Flipkart/Amazon. etc)A brief description of what this project does and who it's for


## Data Discription
The Data was basically ana NLP problem with rating given as a sentiment.
we can clearly see that the scale is upto 5 so we can consider it as a
 classification problem and it is also a supervised machine learning problem.

we are required to awnser the following questions...

    1) Categorize & analyse the reviews to calculate 
    the percentage of positive & negative reviews.
    
    2) Calculate the total rating on a scale of 5 
    for each category.
    
    3) Create a Ranking table for each Mobile phone
    based on each category and overall ranking.


### Features
    'Unnamed: 0', 'Review-Title', 'rating', 'Review-Body', 'Product Name'

The features seems to be very less but the product name column has lot of features to be extracted.
lets drive deep in to the data analysis part.


## Data Cleaning and EDA
### Target variable (ratings)

    5.0 out of 5 stars    9399
    1.0 out of 5 stars    5231
    4.0 out of 5 stars    4886
    3.0 out of 5 stars    2703
    2.0 out of 5 stars    1558

![ratings before clean](https://user-images.githubusercontent.com/98254459/188689659-a81ddb4e-b08d-46d0-b603-8567e20479e8.png)


This is the count of the data. we can cleanly see huge imbalance.

### Product Name

    Redmi 9 Activ (Carbon Black, 4GB RAM, 64GB Storage)           4450
    OPPO A31 (Mystery Black, 6GB RAM, 128GB Storage)              4324
    Redmi 10 Prime (Bifrost Blue 4GB RAM 64GB ROM                 3597
    OnePlus Nord CE 2 5G (Gray Mirror, 8GB RAM, 128GB Storage)    3570
    Redmi Note 11 (Space Black, 4GB RAM, 64GB Storage)            2642
    realme narzo 50A (Oxygen Blue , 4GB RAM + 64 GB Storage)      2100
    vivo iQOO Z6 5G (Chromatic Blue, 6GB RAM, 128GB Storage)      1365
    Samsung Galaxy M32                                            1336

These are the count values. we see that the samsung M32 dose not have any 
clarity on what is the storage and ram size. so we will be droping this feature.

#### colour
the colour feature will be ignored as it is unique among all the phones.

### Target after cleaning.
There was not much change yet after removing Samsung mobiles.
that is also a positive sign.

![ratings aftclean](https://user-images.githubusercontent.com/98254459/188689657-9100ca20-7505-44a2-8027-d1abc757e444.png)




## Splitting the Data
Normally we split the data by train and test but here since we see huge imbalance in the data...
almost 70% and above is an imbalance so we will be balancing the data using the SmoteEnn approach.
this is generally done to clustering approach so the data find correct clusters.
here we are doing as we have KNN.
we have also followed it with other models too.

We have also set the sampling strategy to all.
which means it dose both undersampling and oversampling.
the code is as follows.

    from imblearn.combine import SMOTEENN
    
    smt = SMOTEENN(sampling_strategy='all')
    x_smt, y_smt = smt.fit_resample(x_train, y_train)

## Term Frequency Inverse Document Frequency

The data is mostly NLP based. we have clearly taken the key best features based on the 5 rating classes.
Now the key best words on each row might have words that are very vague (e.g) the name Phone in a phone related data might occur in all the rows.
we have removed this buy the following code.

    tfidf = TfidfVectorizer(max_features = 500)
    


## Models
The Data is a bit large as we know we have used multiple models for this purpose.

Namely: 
KNN, Decision Tree, Random Forest and XG-Boost.

#### KNN:

as we have already clearly done Smote Enn approach to the data.
the model is expected to perform the best. 

the results were also just as expected.
    
    The best score obtained =  1.0
    The best Nearest Neighbors = 4

     [1240    0    0    0    0]
     [   0  333    0    0    0]
     [   0    0  657    0    0]
     [   0    0    0 1139    0]
     [   0    0    0    0 2142]

![KNN](https://user-images.githubusercontent.com/98254459/188689652-dbb1a76e-7992-4ce2-be33-6cfa33d1edde.png)


0 errors.
Still we are not considering this model. 

because, our aim here is not to just find the 5 rating regions but it is also to make the model 
learn where a particular comment might fall.
so understand this the KNN approach just tries to over fit.
we have gained the result neglecting the maximum overfit but still this model is a secondry option.

#### Decision Tree

The Decision Tree also has perfomed well the results are as follows...

    The best score obtained =  0.93
    The best depth = 3

though the score is a bit less. we have the test case to be 100% 
positive a rare case though...

     [1240    0    0    0    0]
     [   0  333    0    0    0]
     [   0    0  657    0    0]
     [   0    0    0 1139    0]
     [   0    0    0    0 2142]

![DT](https://user-images.githubusercontent.com/98254459/188689649-1aa87e59-8dc0-451c-be36-67adc6874310.png)

#### Random forest
This model is a combinations of Decision Trees its on among the Ensemble techniques.
this model has performed great without overfitting. and the score is as follows.

    The best score obtained =  0.99
    The best depth = 14

but in the test we can say this model was not lucky enough.
but still it has performed well.

    [1240    0    0    0    0]
    [   0  312    0    0   21]
    [   0    0  603    0   54]
    [   0    0    0 1055   84]
    [   0    0    0    0 2142]
![RF](https://user-images.githubusercontent.com/98254459/188689662-442001a5-d39b-4c01-9360-e7458d4e3ecb.png)
