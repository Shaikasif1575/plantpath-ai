import urllib.request, json, pathlib, io
from PIL import Image, ImageFilter

results = {}

def post(img_bytes, name_bytes):
    boundary = b'b'
    b = (boundary + b'\r\nContent-Disposition: form-data; name="image"; '
         + b'filename="' + name_bytes + b'"\r\nContent-Type: image/png\r\n\r\n'
         + img_bytes + boundary + b'--')
    req = urllib.request.Request('http://127.0.0.1:8000/api/predict', data=b,
                                 method='POST',
                                 headers={'Content-Type': 'multipart/form-data; boundary=' + boundary.decode()})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())

# Real diseased
img = pathlib.Path('tests/probe_set/author_sample_apple_scab.jpg').read_bytes()
d = post(img, b'a.jpg')
results['real'] = {'detected_disease': d['detected_disease'], 'low_conf': d.get('low_confidence'), 'info_id': d.get('disease_info',{}).get('id')}

# Healthy
healthy = Image.new('RGB', (300,300), (60,140,60))
buf = io.BytesIO()
healthy.save(buf, format='PNG')
dh = post(buf.getvalue(), b'h.png')
results['healthy'] = {'condition': dh['condition'], 'low_conf': dh.get('low_confidence')}

# Red (unrelated)
red = Image.new('RGB', (300,300), (255,30,30))
buf_r = io.BytesIO()
red.save(buf_r, format='PNG')
dr = post(buf_r.getvalue(), b'r.png')
results['red'] = {'low_conf': dr.get('low_confidence'), 'unrelated_warning': any(w.get('code')=='unlikely_leaf' for w in dr.get('validation',{}).get('warnings',[]))}

# Blurry
blurry = Image.new('RGB', (300,300), (50,120,50)).filter(ImageFilter.GaussianBlur(radius=5))
buf_b = io.BytesIO()
blurry.save(buf_b, format='PNG')
dbl = post(buf_b.getvalue(), b'b.png')
results['blurry'] = {'low_conf': dbl.get('low_confidence'), 'blurry_warning': any(w.get('code')=='blurry' for w in dbl.get('validation',{}).get('warnings',[]))}

# Invalid upload
b_bad = b'--b\r\nContent-Disposition: form-data; name="image"; filename="bad.gif"\r\nContent-Type: image/gif\r\n\r\nGIF89a\r\n--b--'
try:
    req_bad = urllib.request.Request('http://127.0.0.1:8000/api/predict', data=b_bad, method='POST', headers={'Content-Type': 'multipart/form-data; boundary=b'})
    urllib.request.urlopen(req_bad, timeout=10)
    results['invalid'] = 'FAIL (no 400)'
except urllib.error.HTTPError as e:
    results['invalid'] = f'PASS (code {e.code})'

# Low confidence contract
results['low_contract'] = 'low_confidence' in d

print("=== CHECKLIST 1 RETEST (updated assertions) ===")
for k,v in results.items():
    print(f"{k}: {v}")
print("\nPASS checklist if:")
print("  real detected_disease='Apple scab'", results['real']['detected_disease'])
print("  healthy condition='healthy'", results['healthy']['condition'])
print("  red has unlikely_leaf warning", results['red']['unrelated_warning'])
print("  blurry has blurry warning", results['blurry']['blurry_warning'])
print("  invalid upload returns 400", results['invalid'])
print("  low_confidence contract present", results['low_contract'])
