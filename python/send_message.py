from twilio.rest import Client
def send_message(action='sit up', to=''):
	account_sid = 'AC025dbe29a3dc777a9433d15edbfbb9ce'
	client = Client(account_sid, auth_token)

        payload = 'The action '+ action + ' was detected.'

	message = client.messages \
			.create(
			     body=payload,
			     from_='+15035641958',
			     to=to
			 )

