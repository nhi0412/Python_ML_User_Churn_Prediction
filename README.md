# Python_ML_User_Churn_Prediction

# **1. Project Introduction**

## **Project Motivation**

In today's competitive business landscape, retaining customers is crucial for long-term growth and profitability. This project aims to develop a predictive model that identifies customers who are at risk of churn and uncover behavioral patterns to enable targeted retention strategies.

## **Business Challenge**

- Patterns and behaviors of churned users.
- Building a Machine Learning Model for predicting churn.
- Segmentation of churned users for effective promotions/campaigns.

## Supervised and Unsupervised Learning in This Project

### 1. Supervised Learning

- **Task**: Predict customer churn (binary classification: churn or not).
- **Label**: 'Churn' column (1 = churned, 0 = active).
- **Goal**: Build a model that learns from labeled historical data to predict churn probability for new customers.

### 2. Unsupervised Learning

- **Task**: Segment churned customers into clusters without using labels.
- **Goal**: Identify groups with similar behaviors (e.g., coupon usage, tenure, order count) to enable personalized marketing strategies.

# **2. Dataset Overview**

| Feature | Description |
| --- | --- |
| CustomerID  | Unique customer ID |
| **Tenure** | Number of months the customer has been active. |
| CityTier | City tier (1,2,3) |
| **WarehouseToHome** | Distance between the warehouse and customer's home. |
| **HourSpendOnApp** | Time spent on the app daily. |
| **OrderCount** | Total number of orders placed. |
| **CouponUsed** | Total number of coupons redeemed. |
| **CashbackAmount** | Total cashback earned. |
| **Complain** | Whether the customer filed a complaint in the last month (1 = yes, 0 = no). |
| **DaySinceLastOrder** | Days since the last order was placed. |
| **SatisfactionScore** | Customer satisfaction rating (1 to 5). |
| PreferredLoginDevice  | Preferred login device of customer |
| **PreferredPaymentMethod** | Preferred method of payment. |
| PreferedOrderCat  | Preferred order category of customer in last month |
| **Gender** | Gender of the customer. |
| NumberOfDeviceRegistered | Total number of devices is registered on particular customer |
| NumberOfAddress  | Total number of added added on particular customer |
| OrderAmountHikeFromLastYear | Percentage increases in order from last year |
| **MaritalStatus** | Marital status (Married/Single). |
| **Churn** | Target variable: whether the customer churned (1) or not (0). |

# 3. Data Preprocessing

## Data Exploration

![Image](https://github.com/user-attachments/assets/8a6e8683-77d2-4cd4-9d98-a8d98bfdfcfd)

![Screenshot 2025-04-27 at 5.37.29 pm.png](attachment:79d067e3-4279-42a0-b879-79916e01dbf3:Screenshot_2025-04-27_at_5.37.29_pm.png)

## **Missing values**

![Screenshot 2025-04-27 at 5.39.05 pm.png](attachment:00f524a0-9ca4-4de0-9be7-777552566086:6d217b6d-6426-4651-a266-1867ab101b0f.png)

![Screenshot 2025-04-27 at 5.40.00 pm.png](attachment:572e41bd-a5d3-4cbc-bd59-871719b4c85d:Screenshot_2025-04-27_at_5.40.00_pm.png)

We impute the missing values using the median, as all columns are numeric and the median is less sensitive to outliers compared to the mean.

## **Duplicate values**

![Screenshot 2025-04-27 at 5.44.04 pm.png](attachment:229e057c-603f-40c0-9292-2f6be2a1b488:Screenshot_2025-04-27_at_5.44.04_pm.png)

There’s no duplicate value in the dataset.

## **Univariate Analysis**

### **Numerical Columns**

![Screenshot 2025-04-27 at 5.48.21 pm.png](attachment:d5e9498e-8b54-4cdf-907b-abf2ed18ac6f:Screenshot_2025-04-27_at_5.48.21_pm.png)

Factors that affect Churn:

**Tenure - Churn:** Strong negative correlation(-0.34): => The longer a customer has been engaged, the less likely they are to churn.

**Complain - Churn:** Light positive correlation (0.25): => Customers who filed a complaint are more likely to churn.

**DaySinceLastOrder - Churn:** Light negative correlation (-0.16): => The more days since the customer's last purchase, the higher the likelihood of churn.

**CashbackAmount - Churn:** Light negative correlation (-0.15): => Customers who received more cashback are less likely to churn.

Other notable correlations:

**OrderCount and CouponUsed:** Strong correlation (0.64) => Customers who use more discount coupons also tend to place more orders.

**DaySinceLastOrder and OrderCount:** Strong correlation (0.42) => High-value customers at risk of churn since their last purchase was a long time ago.

**OrderCount/CouponUsed and CashbackAmount:** Fairly strong correlation (0.32 and 0.22) => More orders/coupons lead to a higher total cashback received.

![Screenshot 2025-04-27 at 5.49.36 pm.png](attachment:3230ef05-1cc4-4bd4-a454-45e3540943a8:Screenshot_2025-04-27_at_5.49.36_pm.png)

![Screenshot 2025-04-27 at 5.51.07 pm.png](attachment:a2394dd7-689e-45f0-a5c2-7b0f5cb06b19:Screenshot_2025-04-27_at_5.51.07_pm.png)

### **Categorical Columns**

![Screenshot 2025-04-27 at 5.53.21 pm.png](attachment:dc473820-27d9-4f9b-af95-738179fc1409:Screenshot_2025-04-27_at_5.53.21_pm.png)

# 4. **Model for predicting churned users**

## **Encoding**

Use **One-hot encoding** to transform categorical features into binary features. Since each column has only a few unique categorical values, one-hot encoding is effective

![Screenshot 2025-04-27 at 5.58.07 pm.png](attachment:9944693d-658e-451d-a0a1-db0d12daba42:Screenshot_2025-04-27_at_5.58.07_pm.png)

## **Apply model**

### **Split/Train/Validate/Test set**

![Screenshot 2025-04-27 at 6.00.51 pm.png](attachment:6da0cc32-b567-4018-acb9-ad848e576f8e:Screenshot_2025-04-27_at_6.00.51_pm.png)

### **Apply Model**

Random Forest was used to build the churn prediction model because it is highly effective for binary classification problems and handles complex, non-linear relationships well. It also provides feature importance rankings, offering valuable insights into which factors contribute most to customer churn.

![Screenshot 2025-04-27 at 6.01.41 pm.png](attachment:cdb5cfc9-74a6-4ef5-9ab6-ffb953b2e697:9e5fe996-51d2-47fa-8bf2-61aa4a0f0891.png)

- The training accuracy is 100% => model fits perfectly on the training data.
- The testing accuracy is high (94.14%)=> model generalizes well to unseen data.
- There is no severe overfitting as the gap between training and testing accuracy is not too large.

![Screenshot 2025-04-27 at 6.02.22 pm.png](attachment:fd86f046-a5bf-4dc7-a108-c9dd87ab0cc5:Screenshot_2025-04-27_at_6.02.22_pm.png)

- The balanced accuracy on the test set is 85.68% => model performs well even if the classes are imbalanced

### **Feature Important**

![Screenshot 2025-04-27 at 6.04.16 pm.png](attachment:e38bad8a-d132-4c3f-819a-cc1cee633024:Screenshot_2025-04-27_at_6.04.16_pm.png)

Tenure is the strongest factor influencing churn (~0.20), followed by CashbackAmount and WarehouseToHome(likely related to delivery speed)

**Top 5 factors that influence churn rate:**

1. Tenure
2. CashbackAmount
3. WarehouseToHome
4. Complain
5. DaySinceLastOrder

**Tenure Aspect:**

![Screenshot 2025-04-27 at 6.05.08 pm.png](attachment:ca255455-c6d9-4167-9b62-25bfc024ecf7:Screenshot_2025-04-27_at_6.05.08_pm.png)

- Nearly 350 churned users had a tenure of less than 2 months.
- A small number of users churn even after 9 months, but this group is negligible compared to early churn.

Action Plan:

- Implement a strong onboarding experience within the first 60 days.
- Introduce welcome bonuses and early engagement incentives.
- Develop loyalty programs targeting users who pass the 6-month and 12-month marks to reinforce long-term retention.

**CashbackAmount Aspect:**

![Screenshot 2025-04-27 at 6.06.34 pm.png](attachment:69b18e2e-f718-4f85-a0d3-66860d32df62:Screenshot_2025-04-27_at_6.06.34_pm.png)

- Majority of churned users had CashbackAmount between 120–200.
- Higher cashback amounts (>250) are associated with lower churn.

Action Plan:

- Increase cashback incentives for users with low spend or initial transactions.
- Focus on the low cashback users with better loyalty program/tiers
- Design loyalty tiers where cashback percentages increase over time to encourage longer retention.

**WarehouseToHome Aspect:**

![Screenshot 2025-04-27 at 6.06.47 pm.png](attachment:ce9f5910-7831-4e2a-b58a-89fdaafba1fe:Screenshot_2025-04-27_at_6.06.47_pm.png)

- Common distance ranges from 5-16km (the highest point is 14km)
- The range of distance for churned users is from 5-35km
- Churn is not limited to customers far from the warehouse but delivery time and customer service.

Action Plan:

- Improve delivery speed and communication, especially for nearby users.
- Implement real-time delivery tracking and proactive communication during delays.
- Conduct customer satisfaction surveys to identify pain points in the delivery process.

**Complain Aspect:**

![Screenshot 2025-04-27 at 6.06.58 pm.png](attachment:fdcd9053-4779-4130-b467-c8bee3bf70e9:Screenshot_2025-04-27_at_6.06.58_pm.png)

- Churn is higher among users who complained.
- However, silent churn (no complaints) is also significant.

Action Plan:

- Strengthen customer service and speed up complaint resolution.
- Implement proactive customer feedback surveys to detect dissatisfaction early.

**DaySinceLastOrder Aspect:**

![Screenshot 2025-04-27 at 6.07.15 pm.png](attachment:8f27c52d-5bb8-4432-9c89-cc47ab1a539f:Screenshot_2025-04-27_at_6.07.15_pm.png)

- Recent buyers (within 10 days) still churn, suggesting post-purchase experience issues.
- Lack of activity beyond 10 days is an early indicator of churn.

Action Plan:

- Enhance during and post-purchase support and engagement (delivery, feedback, tracking, feedback, refund)
- Implement reactivation campaigns targeting users inactive for over 7 days.

### **Model evaluation - Fine-tuning**

Hyperparameter Tuning

![Screenshot 2025-04-27 at 6.09.08 pm.png](attachment:28d9cbd4-b73e-4969-a800-05b1c1f28136:3c1dfcf5-be96-4964-80b6-ef4056bc49bf.png)

After hyperparameter tuning, the model was optimized with the following parameters: **{'bootstrap': False, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}.**

As a result, the fine-tuned model achieved:

- **Test set accuracy: 0.9526**
- **Balanced test accuracy: 0.8883**

Both the overall accuracy and the balanced accuracy are higher compared to the baseline model, indicating that hyperparameter tuning has significantly improved the model's performance.

# 4. **Model for predicting churned users**

![Screenshot 2025-04-27 at 6.13.39 pm.png](attachment:36999b2a-d5cb-4e09-815d-766d2eaf365e:Screenshot_2025-04-27_at_6.13.39_pm.png)

Filter out one churned user.

## **Encoding**

Use **One-hot encoding** to transform categorical features into binary features. Since each column has only a few unique categorical values, one-hot encoding is effective

![Screenshot 2025-04-27 at 6.16.12 pm.png](attachment:6f6441af-bfb4-4b8a-a10e-7488f1c14a53:Screenshot_2025-04-27_at_6.16.12_pm.png)

## **Dimension Reduce**

![Screenshot 2025-04-27 at 6.16.29 pm.png](attachment:442acbe4-ec89-4088-8632-ca37ff325b25:Screenshot_2025-04-27_at_6.16.29_pm.png)

![Screenshot 2025-04-27 at 6.16.38 pm.png](attachment:97020198-9a31-4824-adef-a1db751ec86e:Screenshot_2025-04-27_at_6.16.38_pm.png)

Reduce Dimension to 3 columns:

- Col1 has **99.94%** variance.
- Col1 has **0.05%** variance
- Col1 has **0.0025%** variance

## Find K value

![Screenshot 2025-04-27 at 6.20.36 pm.png](attachment:0bc35bd4-38f5-4c85-b219-7d9f08481c06:Screenshot_2025-04-27_at_6.20.36_pm.png)

⇒ k=3

## **Apply K-Means**

K-Means clustering was used because there were no group labels, making it an unsupervised learning task. The goal was to uncover hidden patterns within churned customers' behavior, such as differences in coupon usage, tenure, or purchasing activity. 

K-Means is an efficient and scalable algorithm that groups similar users based on their behavior, enabling the company to design targeted, personalized re-engagement strategies for each customer segment. This combination of predictive modeling and customer segmentation provides a comprehensive approach to reducing churn and improving customer retention.

![Screenshot 2025-04-27 at 6.23.15 pm.png](attachment:9df639a4-00b2-40d0-9477-b4813603e6fe:Screenshot_2025-04-27_at_6.23.15_pm.png)

![Screenshot 2025-04-27 at 6.22.46 pm.png](attachment:d35276a5-1f03-4a86-857e-be810d06664d:Screenshot_2025-04-27_at_6.22.46_pm.png)

## **Evaluate Model**

### **Silhouette Score**

![Screenshot 2025-04-27 at 6.23.59 pm.png](attachment:2282f74e-83f6-4a82-8ccb-6448767f13b2:Screenshot_2025-04-27_at_6.23.59_pm.png)

The Silhouette Score of 0.5933 indicates that the clustering structure is fairly strong, with reasonably well-separated and cohesive clusters.

## **Feature Important**

![Screenshot 2025-04-27 at 6.24.57 pm.png](attachment:02b97a09-ff72-49ae-a6de-f6611e28ef54:Screenshot_2025-04-27_at_6.24.57_pm.png)

CouponUsed is the strongest factor influencing churn (~0.12), followed by Tenure, OrderCount and CashbackAmount

**Top 4 factors that influence churn rate:**

1. CouponUsed
2. Tenure
3. OrderCount
4. CashbackAmount

**CouponUsed Aspect:**

![Screenshot 2025-04-27 at 6.25.53 pm.png](attachment:78525220-18b4-46ba-a36a-a72ff76da86a:Screenshot_2025-04-27_at_6.25.53_pm.png)

- Cluster 1: Median coupon usage is the lowest (near 0). Most customers rarely use discount coupons => customers less dependent on promotions.
- Clusters 0 and 2: Higher median coupon usage. More outliers (some customers used more than 10 coupons) => more likely to use coupons

Action Plan:

- Low coupon usage (Cluster 1): Focus on non-price benefits (service quality, loyalty perks). Communicate brand value instead of promotions.
- Higher coupon usage (Clusters 0 and 2): Use targeted discount campaigns (limited-time, new-user discounts) to promote purchases. Offer graduated discount plans (e.g., 10% for first order, 5% for next.

**Tenure Aspect:**

![Screenshot 2025-04-27 at 6.26.35 pm.png](attachment:a91c4a34-3670-4192-a15d-fb86b1aacffa:Screenshot_2025-04-27_at_6.26.35_pm.png)

- Cluster 0: Low median (1–2 months). Mostly newer customers, a few longer-term outliers => new customers
- Cluster 1: Widest range tenure (0–9 months), with many customers having stayed for 20 months => loyal customers with long engagement
- Cluster 2: Low median (1–2 months). Majority are very new customers (2-3 months) => extremely new customers

Action Plan:

- Cluster 1: Launch loyalty reward programs (points, cashback after X months). Send personalized thank-you emails or gifts after (6 months or 12 months), early access to new products, premium service.
- Cluster 0 and 2: Set up welcome series campaigns (email or in-app messages). Offer progressive incentives (e.g., small discounts after first purchase, bigger rewards after second or third), Educate about long-term value through onboarding content.

**OrderCount Aspect:**

![Screenshot 2025-04-27 at 6.26.50 pm.png](attachment:27d332ef-64e0-46e6-9de2-34d2e80c0836:Screenshot_2025-04-27_at_6.26.50_pm.png)

- Cluster 1: Has the lowest median number of orders (~1 order). Narrow distribution, most customers placed only 1–2 orders => very few purchases
- Clusters 0 and 2: Higher medians (2–3 orders). Wider spread for cluster 0, some customers placed up to 10–15 orders => more active, with more frequent orders

Action Plan:

- Cluster 1: Reactivation campaigns offers easy reorder options. Highlight new products or bundles to draw interest. Use loyalty challenges ("Order twice this month, get a bonus").
- Clusters 0 and 2: Reward consistency: encourage repeat orders via loyalty points. Recommend related products, bundles.

**CashbackAmount Aspect:**

![Screenshot 2025-04-27 at 6.27.09 pm.png](attachment:9308cedc-cda6-4229-8227-7e81192b1577:Screenshot_2025-04-27_at_6.27.09_pm.png)

- Cluster 1: Lower median cashback amount compared to other clusters. Cashback mostly stays around 120–150 units.
- Clusters 0 and 2: Higher medians (~160–170 units). Wider distribution and more customers received higher cashback rewards (>250 units).

Action Plan:

- Cluster 1: Offer targeted cashback promotions to re-engage. Design first-purchase cashback offers to motivate another order.
- Clusters 0 and 2: Maintain engagement through progressive rewards and celebrate loyalty (Top 10% cashback earners)
