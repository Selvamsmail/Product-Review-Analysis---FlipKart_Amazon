
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

![ratings before clean](https://user-images.githubusercontent.com/98254459/189052656-95d5bf02-40c6-4666-8728-976f1cf12e20.png)



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

### Feature Engneering.
we have added 2 new features from the product names of the phones.
the features were namely.
* RAM
* Storage.
it belongs to the quality defination of the phone so the model could learn something.
it was an additional quality defination part of the phone.


### Target after cleaning.
There was not much change yet after removing Samsung mobiles.
that is also a positive sign.

![ratings aftclean](https://user-images.githubusercontent.com/98254459/189052653-a3874c9b-9057-4ba1-a8e2-18b89785bfa6.png)





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

![KNN](https://user-images.githubusercontent.com/98254459/189052649-793099fb-fb7e-4eb2-8f36-01c5bb4b7f03.png)


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

![DT](https://user-images.githubusercontent.com/98254459/189052645-fd2dc46c-732e-4d93-9320-11e075786ce4.png)


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
![RF](https://user-images.githubusercontent.com/98254459/189052659-9a4e24d7-fd91-4622-82e9-05d7f4f26e75.png)

## Insights and Questions
Since the data is NLP oriented we dont have much features to gather insights on.
but we can definately throw light on certain factors like Ram and Storage.

we also have questions to be answered.
1. Categorize & analyse the reviews to calculate the percentage of positive & negative reviews.


        for this we already have the ratings feature.
        we will also plot word clouds for the key features.
    
    ![Sentiment percent](https://user-images.githubusercontent.com/98254459/189052664-85e778a7-7fb2-4150-a333-813b58ba13db.png)


    we can see that there is not much difference there are equal ammount of positive and negative reviews with 20% difference.

#### Wordcloud

![wordcloud](https://user-images.githubusercontent.com/98254459/189052669-b7357f92-a13a-4859-8b83-484c7a773a93.png)


As we already know the word phone in phones dataset is meaning less. we have done TF-IDF but it has been done
while we turned the key words to features.
so we have not plotted after the extraction.

2. Calculate the total rating on a scale of 5 for each category.

    The question was not very clear. but based on the features we have...

    'Review-Title', 'rating', 'Review-Body', 'Product Name'
    
    only product name is left for analysis.
    so we are to classify name wise ratings.
    and the plot is as follows.

    ![classwise rating](https://user-images.githubusercontent.com/98254459/189052640-935a00c8-d387-4f63-b053-058d35d5e2b8.png)

3. Create a Ranking table for each Mobile phone based on each category and overall ranking.

    The only numeric feature was rank. so to create ranking table we have used df.rank() command...

    the plot of the best phone based on the user rating count is as follows...

    ![Rank](https://user-images.githubusercontent.com/98254459/189052651-5ce8d35a-b7fd-4691-831f-7615ce6b9180.png)





## Conclusion
The data was a bit large so the XG boost model was not able to compile with 12 gb of ram.
but we have seen that the other models have performed well and we had 60% positive reviews 
but all phones also had equal ammount of negative reviews.
which is to be considered and based on this the better performing phones are alone mentioned in the
above plots.

through the rating scale of each phones we can definately gather a lot of insights. that helps the companies understand their
position in the market but appart from that we dont have much to analyze from a NLP data.