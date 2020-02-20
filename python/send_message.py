from twilio.rest import Client
import boto3
def send_message(action='sit up', to='+18585006186'):
    phone_numbers= {'brandon':'+15712513711','ji':'+18585006186','ricky':'+18307198702'}
    #'george':'+18586995436'
    client = boto3.client('s3', region_name='us-west-2')
    client.upload_file('python/detection_images/save_image.png', 'clientimagedata', 'save_image.png', ExtraArgs={'ACL':'public-read', 'ContentType':'image/png'})
    print('uploaded')
    image_path = 'https://clientimagedata.s3-us-west-2.amazonaws.com/save_image.png'
    #image_path = 'https://c1.staticflickr.com/3/2899/14341091933_1e92e62d12_b.jpg'

    client = Client(account_sid, auth_token)

    payload = 'Ricky has sat up out of bed. Please provide assistance.'

    for k in phone_numbers:
        to = phone_numbers[k]
        #message = client.messages \
            #.create(
                 #body=payload,
                 #from_='+18582391987',
                 #to=to
             #)


        #print('sent message')
        message = client.messages \
            .create(
            body=payload,
            from_='+18582391987',
            media_url=[image_path],
            to=to
            )
        print('sent image')
        print(message.sid)


