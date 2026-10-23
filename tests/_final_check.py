import urllib.request, json, pathlib
img = pathlib.Path('tests/probe_set/author_sample_apple_scab.jpg').read_bytes()
b = (b'--b\r\nContent-Disposition: form-data; name="image"; '
     b'filename="a.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'
     + img + b'\r\n--b--')
req = urllib.request.Request('http://127.0.0.1:8000/api/predict', data=b, method='POST',
                             headers={'Content-Type': 'multipart/form-data; boundary=b'})
r = urllib.request.urlopen(req, timeout=30)
d = json.loads(r.read())
print("detected_disease:", d['detected_disease'])
print("condition:", d['condition'])
print("disease_info id:", d.get('disease_info', {}).get('id') if d.get('disease_info') else 'NONE')
