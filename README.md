# DataScience-Coursework-2
<B>ABSTRACT:</B>
<BR>
  <B>Prediction of car price using kaggle car dataset. </B>
<BR>
  

<BR>

 The prediction of price of a car based on its various features available in dataset such as kilometres used, year of buying, Car name, fuel type etc.
<BR>
  We will be working on a used Vehicle dataset from cardekho Dataset . This dataset contains information about used cars listed on www.cardekho.com (Kaggle.com, 2018). We are going to use for finding predictions of price with the use of regression models.
Various features and their meanings:
1.	Car_Name: Car name. 

2.	Year: Car year when its originally brought. 

3.	Selling_Price: Owner value which the need to sell the vehicle 

4.	Present_Price: Showroom cost of the vehicle particularly ex-showroom cost. 

5.	Kms_Driven: Covered separation via vehicle in km . 

6.	Fuel_Type: Car fuel type diesel or petroleum. 

7.	Seller_Type: Describe the vehicle vender is a seller or a person. 

8.	Transmission: Describe its usefulness either vehicle is manual or programmed. 

9.	Owner: The proprietors of the past vehicle.

  
<BR>
 ![image](https://user-images.githubusercontent.com/103975775/173398754-2ff85eb6-6a13-4841-bb34-7a764c5cf708.png)

<br>
<BR>
    <B> TASK INCLUDE: </B>
              1. Importing and Loading Dataset into Spyder  
              2. Cleaning the Data, Dealing with missing Values
              3. Visualization of features
              4. Slicing 
              5. Model Prepare 
              6. Tuning
              7. Model comparision
  
<BR>
<BR>
<b>Reduction of Selling price w.r.t Present Showroom Prices:</b>
 
Here, we can find following observations:
1.	The cars with a greater number of owners are having large reduction of Selling Price.
2.	Cars with Fuel Type-1(i.e. Diesel Cars) have higher depreciation in Selling Price.
3.	Seller Type-0(i.e. Cars sold from Dealers have higher depreciation in Selling Price.
4.	The cars with automatic transmission(i.e. value=1) have higher depreciation in Selling Price.

<BR>
<br>
  ![image](https://user-images.githubusercontent.com/103975775/173398395-bb5b1314-25b9-4880-ad3f-faa3abf19520.png)
 <BR>
 <BR>
   <b> Results: MODEL COMPARISION </b>
 <BR>  
 <BR>
<BR>
![image](https://user-images.githubusercontent.com/103975775/173399449-49981fc4-f9bf-4826-bb44-265442fe2fe9.png)
<BR>
  
<BR>
  ![image](https://user-images.githubusercontent.com/103975775/173399387-aaba08f0-f82b-4baa-93af-0f2793f536fe.png)
<BR>
  
<BR>  
 <B> Based on the final data frame, it gives opinion about the score of models and also the plots help us to understand which models is more successful. 
   Here we are getting better results from Random Forest Regressor Algorithm with least value of RMSE error. 
   As a result,  after comparison of different models we can say Random Forest Regression model is the best model to predict the car selling price by accuracy of 95%. <B>
   
<br>
   THANKS
  
