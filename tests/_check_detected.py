import urllib.request, json, pathlib
img = pathlib.Path('tests/probe_set/author_sample_apple_scab.jpg').read_bytes()
boundary = 'b'
b = (b'--' + boundary.encode() + b'\r\nContent-Disposition: form-data; name="image"; '
     b'filename="a.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'
     + img + b'\r\n--' + boundary.encode() + b'--')
req = urllib.request.Request('http://127.0.0.1:8000/api/predict', data=b,
                             method='POST',
                             headers={'Content-Type': 'multipart/form-data; boundary=' + boundary})
r = urllib.request.urlopen(req, timeout=30)
data = json.loads(r.read())
print("detected_disease:", data.get('detected_disease'))
print("disease:", data.get('disease'))
