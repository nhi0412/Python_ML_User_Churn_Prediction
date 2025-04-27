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

![Image](https://github.com/user-attachments/assets/0b7bd5bc-a16d-4c1f-9cd3-a97fab5ee2e4)

## **Missing values**

![Image](https://github.com/user-attachments/assets/4ced4327-d674-4010-a227-adb600dbf3c7)

![Image](https://github.com/user-attachments/assets/834d1da0-f0d9-4a4a-b05d-58e8ace5a62b)

We impute the missing values using the median, as all columns are numeric and the median is less sensitive to outliers compared to the mean.

## **Duplicate values**

![Image](https://github.com/user-attachments/assets/20d291df-845c-4a3b-9ae0-238a4cf907b7)

There’s no duplicate value in the dataset.

## **Univariate Analysis**

### **Numerical Columns**

![Image](https://github.com/user-attachments/assets/0c0c1b12-7532-497a-8417-da57f9bb9d06)

Factors that affect Churn:

**Tenure - Churn:** Strong negative correlation(-0.34): => The longer a customer has been engaged, the less likely they are to churn.

**Complain - Churn:** Light positive correlation (0.25): => Customers who filed a complaint are more likely to churn.

**DaySinceLastOrder - Churn:** Light negative correlation (-0.16): => The more days since the customer's last purchase, the higher the likelihood of churn.

**CashbackAmount - Churn:** Light negative correlation (-0.15): => Customers who received more cashback are less likely to churn.

Other notable correlations:

**OrderCount and CouponUsed:** Strong correlation (0.64) => Customers who use more discount coupons also tend to place more orders.

**DaySinceLastOrder and OrderCount:** Strong correlation (0.42) => High-value customers at risk of churn since their last purchase was a long time ago.

**OrderCount/CouponUsed and CashbackAmount:** Fairly strong correlation (0.32 and 0.22) => More orders/coupons lead to a higher total cashback received.

![Image](https://github.com/user-attachments/assets/e383e71c-766d-40dd-a1b6-447d0d455e1a)
![Image](https://github.com/user-attachments/assets/7f97ff43-dec1-41bd-9e54-5747b898fc8f)

### **Categorical Columns**

![Image](https://github.com/user-attachments/assets/7c0cecb2-5d17-48a4-abf9-eb32a703ba28)

# 4. **Model for predicting churned users**

## **Encoding**

Use **One-hot encoding** to transform categorical features into binary features. Since each column has only a few unique categorical values, one-hot encoding is effective

![Image](https://github.com/user-attachments/assets/8547ca80-0cb8-4eaa-8438-8e2ae4cf946d)

## **Apply model**

### **Split/Train/Validate/Test set**

![Image](https://github.com/user-attachments/assets/4311a533-80ae-4fbc-876a-3ec58b561f67)

### **Apply Model**

Random Forest was used to build the churn prediction model because it is highly effective for binary classification problems and handles complex, non-linear relationships well. It also provides feature importance rankings, offering valuable insights into which factors contribute most to customer churn.

![Image](https://github.com/user-attachments/assets/516911e3-f869-4508-b581-73781ae11e11)

- The training accuracy is 100% => model fits perfectly on the training data.
- The testing accuracy is high (94.14%)=> model generalizes well to unseen data.
- There is no severe overfitting as the gap between training and testing accuracy is not too large.

![Image](https://github.com/user-attachments/assets/56b37c49-30ce-4592-bc19-6d677ec378f2)

- The balanced accuracy on the test set is 85.68% => model performs well even if the classes are imbalanced

### **Feature Important**

![Image](https://github.com/user-attachments/assets/496beeaa-f9d0-471e-8020-af0c719b9340)

Tenure is the strongest factor influencing churn (~0.20), followed by CashbackAmount and WarehouseToHome(likely related to delivery speed)

**Top 5 factors that influence churn rate:**

1. Tenure
2. CashbackAmount
3. WarehouseToHome
4. Complain
5. DaySinceLastOrder

**Tenure Aspect:**

![Image](https://github.com/user-attachments/assets/2c33b0c5-5540-4efc-a56f-8a558208a2b1)

- Nearly 350 churned users had a tenure of less than 2 months.
- A small number of users churn even after 9 months, but this group is negligible compared to early churn.

Action Plan:

- Implement a strong onboarding experience within the first 60 days.
- Introduce welcome bonuses and early engagement incentives.
- Develop loyalty programs targeting users who pass the 6-month and 12-month marks to reinforce long-term retention.

**CashbackAmount Aspect:**

![Image](https://github.com/user-attachments/assets/f1e8bbfe-eb3c-4638-be30-6ec4e94b8768)

- Majority of churned users had CashbackAmount between 120–200.
- Higher cashback amounts (>250) are associated with lower churn.

Action Plan:

- Increase cashback incentives for users with low spend or initial transactions.
- Focus on the low cashback users with better loyalty program/tiers
- Design loyalty tiers where cashback percentages increase over time to encourage longer retention.

**WarehouseToHome Aspect:**

![Image](https://github.com/user-attachments/assets/1fbe23b8-f6c7-48ef-9940-76c59146e02a)

- Common distance ranges from 5-16km (the highest point is 14km)
- The range of distance for churned users is from 5-35km
- Churn is not limited to customers far from the warehouse but delivery time and customer service.

Action Plan:

- Improve delivery speed and communication, especially for nearby users.
- Implement real-time delivery tracking and proactive communication during delays.
- Conduct customer satisfaction surveys to identify pain points in the delivery process.

**Complain Aspect:**

![Image](https://github.com/user-attachments/assets/1f247b56-222d-40aa-b7b1-ad236893bca6)

- Churn is higher among users who complained.
- However, silent churn (no complaints) is also significant.

Action Plan:

- Strengthen customer service and speed up complaint resolution.
- Implement proactive customer feedback surveys to detect dissatisfaction early.

**DaySinceLastOrder Aspect:**

![Image](https://github.com/user-attachments/assets/272cfe04-98b0-450d-a0c9-59eba955f8b5)

- Recent buyers (within 10 days) still churn, suggesting post-purchase experience issues.
- Lack of activity beyond 10 days is an early indicator of churn.

Action Plan:

- Enhance during and post-purchase support and engagement (delivery, feedback, tracking, feedback, refund)
- Implement reactivation campaigns targeting users inactive for over 7 days.

### **Model evaluation - Fine-tuning**

Hyperparameter Tuning

![Image](https://github.com/user-attachments/assets/91278b20-2f21-4911-8e08-98e3ffc0c00a)

After hyperparameter tuning, the model was optimized with the following parameters: **{'bootstrap': False, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}.**

As a result, the fine-tuned model achieved:

- **Test set accuracy: 0.9526**
- **Balanced test accuracy: 0.8883**

Both the overall accuracy and the balanced accuracy are higher compared to the baseline model, indicating that hyperparameter tuning has significantly improved the model's performance.

# 4. **Model for predicting churned users**

![Image](https://github.com/user-attachments/assets/d748b084-9539-476d-94e4-76469f31b5f6)

Filter out one churned user.

## **Encoding**

Use **One-hot encoding** to transform categorical features into binary features. Since each column has only a few unique categorical values, one-hot encoding is effective

![Image](https://github.com/user-attachments/assets/25b54555-5530-4999-ae48-0a2595c10264)

## **Dimension Reduce**

![Image](https://github.com/user-attachments/assets/85b9b0df-c701-4274-b312-f10488af818e)

![Image](https://github.com/user-attachments/assets/273131c0-1b59-415c-87d7-e5072e4d9f46)

Reduce Dimension to 3 columns:

- Col1 has **99.94%** variance.
- Col1 has **0.05%** variance
- Col1 has **0.0025%** variance

## Find K value

![Image](https://github.com/user-attachments/assets/fabeadf5-3482-4669-ac1f-1eb5cd1a8364)

⇒ k=3

## **Apply K-Means**

K-Means clustering was used because there were no group labels, making it an unsupervised learning task. The goal was to uncover hidden patterns within churned customers' behavior, such as differences in coupon usage, tenure, or purchasing activity. 

K-Means is an efficient and scalable algorithm that groups similar users based on their behavior, enabling the company to design targeted, personalized re-engagement strategies for each customer segment. This combination of predictive modeling and customer segmentation provides a comprehensive approach to reducing churn and improving customer retention.

![Image](https://github.com/user-attachments/assets/29ea7bfd-a70d-4755-adf5-773e76a7729a)

![Image](https://github.com/user-attachments/assets/42679357-5d5e-432e-b05e-33b7084f7af6)

## **Evaluate Model**

### **Silhouette Score**

![Image](https://github.com/user-attachments/assets/9a6709b2-40cb-4e7a-b27a-ea12f36da2ac)

The Silhouette Score of 0.5933 indicates that the clustering structure is fairly strong, with reasonably well-separated and cohesive clusters.

## **Feature Important**

![Image](https://github.com/user-attachments/assets/edc540eb-9396-4e0d-9d37-dd120176e7a4)

CouponUsed is the strongest factor influencing churn (~0.12), followed by Tenure, OrderCount and CashbackAmount

**Top 4 factors that influence churn rate:**

1. CouponUsed
2. Tenure
3. OrderCount
4. CashbackAmount

**CouponUsed Aspect:**

![Image](https://github.com/user-attachments/assets/306257cf-2d08-48ff-9a7a-a6af2847cb57)

- Cluster 1: Median coupon usage is the lowest (near 0). Most customers rarely use discount coupons => customers less dependent on promotions.
- Clusters 0 and 2: Higher median coupon usage. More outliers (some customers used more than 10 coupons) => more likely to use coupons

Action Plan:

- Low coupon usage (Cluster 1): Focus on non-price benefits (service quality, loyalty perks). Communicate brand value instead of promotions.
- Higher coupon usage (Clusters 0 and 2): Use targeted discount campaigns (limited-time, new-user discounts) to promote purchases. Offer graduated discount plans (e.g., 10% for first order, 5% for next.

**Tenure Aspect:**

![Image](https://github.com/user-attachments/assets/326abe9d-aac8-43d7-9df4-a2338bd38374)

- Cluster 0: Low median (1–2 months). Mostly newer customers, a few longer-term outliers => new customers
- Cluster 1: Widest range tenure (0–9 months), with many customers having stayed for 20 months => loyal customers with long engagement
- Cluster 2: Low median (1–2 months). Majority are very new customers (2-3 months) => extremely new customers

Action Plan:

- Cluster 1: Launch loyalty reward programs (points, cashback after X months). Send personalized thank-you emails or gifts after (6 months or 12 months), early access to new products, premium service.
- Cluster 0 and 2: Set up welcome series campaigns (email or in-app messages). Offer progressive incentives (e.g., small discounts after first purchase, bigger rewards after second or third), Educate about long-term value through onboarding content.

**OrderCount Aspect:**

![Image](https://github.com/user-attachments/assets/0ad3fae6-25e4-46b8-8a4e-06213cb983d0)

- Cluster 1: Has the lowest median number of orders (~1 order). Narrow distribution, most customers placed only 1–2 orders => very few purchases
- Clusters 0 and 2: Higher medians (2–3 orders). Wider spread for cluster 0, some customers placed up to 10–15 orders => more active, with more frequent orders

Action Plan:

- Cluster 1: Reactivation campaigns offers easy reorder options. Highlight new products or bundles to draw interest. Use loyalty challenges ("Order twice this month, get a bonus").
- Clusters 0 and 2: Reward consistency: encourage repeat orders via loyalty points. Recommend related products, bundles.

**CashbackAmount Aspect:**

![Image](https://github.com/user-attachments/assets/0ad3fae6-25e4-46b8-8a4e-06213cb983d0)

- Cluster 1: Lower median cashback amount compared to other clusters. Cashback mostly stays around 120–150 units.
- Clusters 0 and 2: Higher medians (~160–170 units). Wider distribution and more customers received higher cashback rewards (>250 units).

Action Plan:

- Cluster 1: Offer targeted cashback promotions to re-engage. Design first-purchase cashback offers to motivate another order.
- Clusters 0 and 2: Maintain engagement through progressive rewards and celebrate loyalty (Top 10% cashback earners)
