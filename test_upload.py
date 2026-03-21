import urllib.request
import urllib.parse
import json

boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = (
    '--' + boundary + '\r\n'
    'Content-Disposition: form-data; name="relative_path"\r\n\r\n'
    'test.heic\r\n'
    '--' + boundary + '\r\n'
    'Content-Disposition: form-data; name="image"; filename="test.heic"\r\n'
    'Content-Type: image/heif\r\n\r\n'
).encode('utf-8')

with open('test.heic', 'rb') as f:
    body += f.read()

body += ('\r\n--' + boundary + '--\r\n').encode('utf-8')

req = urllib.request.Request('http://127.0.0.1:8000/api/register-image', data=body)
req.add_header('Content-Type', 'multipart/form-data; boundary=' + boundary)

try:
    with urllib.request.urlopen(req) as response:
        print(response.getcode())
        print(response.read().decode('utf-8')[:200])
except urllib.error.HTTPError as e:
    print('HTTP Error:', e.code)
    print(e.read().decode('utf-8'))
except Exception as e:
    print('Error:', e)
