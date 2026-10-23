import urllib.request, json, pathlib
img = pathlib.Path('tests/probe_set/author_sample_apple_scab.jpg').read_bytes()
b = (b'--b\r\nContent-Disposition: form-data; name="image"; '
     b'filename="a.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'
     + img + b'\r\n--b--')
req = urllib.request.Request(
    'http://127.0.0.1:8000/api/predict',
    data=b, method='POST',
    headers={'Content-Type': 'multipart/form-data; boundary=b'})
r = urllib.request.urlopen(req, timeout=30)
data = json.loads(r.read())
info = data.get('disease_info')
print('status:', data['status'], '| demo:', data['demo_mode'])
if info:
    print('id:', info['id'])
    print('treatment:', len(info.get('treatment', [])))
    print('pesticides:', len(info.get('pesticides', [])))
    print('sources:', info.get('sources', [])[:1])
    print('disclaimer len:', len(info.get('disclaimer', '')))
    print('has_ingredients:', bool(info.get('pesticides', [{}])[0].get('active_ingredients')))
else:
    print('disease_info: NONE (unexpected!)')
