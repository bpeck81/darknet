from twilio.rest import Client
def send_message(action='sit up', to='+15712513711'):
	account_sid = 'AC025dbe29a3dc777a9433d15edbfbb9ce'
	auth_token = '6f31f4a09df1eff674c7a78a2c5e89c8'
	client = Client(account_sid, auth_token)

	payload = 'The action '+ action + ' was detected.'

	message = client.messages \
			.create(
			     body=payload,
			     from_='+15035641958',
			     to=to
			 )

