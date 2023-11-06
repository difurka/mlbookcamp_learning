

import requests

url = 'http://localhost:9696/predict'

customer = {
    'ClientPeriod': 10000,
    'MonthlySpending': 10000,
    'TotalSpent': 0,
    'Sex': 'Male',
    'IsSeniorCitizen': 0,
    'HasPartner': 'No',
    'HasChild': 'No',
    'HasPhoneService': 'No',
    'HasMultiplePhoneNumbers': 'No',
    'HasInternetService': 'No',
    'HasOnlineSecurityService': 'No internet service',
    'HasOnlineBackup': 'No internet service',
    'HasDeviceProtection': 'No internet service',
    'HasTechSupportAccess': 'No internet service',
    'HasOnlineTV': 'No internet service',
    'HasMovieSubscription': 'No internet service',
    'HasContractPhone': 'One year',
    'IsBillingPaperless': 'Yes',
    'PaymentMethod': 'Mailed check'
}


response = requests.post(url, json=customer).json()
print(response)

if response['churn'] == True:
    print('sending promo email')
else:
    print('not sending promo email')