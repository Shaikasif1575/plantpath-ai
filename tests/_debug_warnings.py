import urllib.request, json, pathlib, io
from PIL import Image, ImageFilter

def request(image_bytes, filename):
    b = (b'--b\r\nContent-Disposition: form-data; name="image"; '
         b'filename="' + filename + b'"\r\nContent-Type: image/png\r\n\r\n'
         + image_bytes + b'\r\n--b--')
    req = urllib.request.Request('http://127.0.0.1:8000/api/predict', data=b,
                                 method='POST',
                                 headers={'Content-Type': 'multipart/form-data; boundary=b'})
    r = urllib.request.urlopen(req, timeout=30)
    return json.loads(r.read())

# Red (unrelated)
red = Image.new('RGB', (300, 300), (255, 30, 30)).filter(ImageFilter.GaussianBlur(radius=2))
buf_r = io.BytesIO()
red.save(buf_r, format='PNG')
red_bytes = buf_r.getvalue()
red_result = request(red_bytes, b'red.png')
print("RED RESULT validation:", red_result.get('validation'))
warnings = red_result.get('validation', {}).get('warnings', [])
print("RED warnings:", warnings)
print("RED has unlikely_leaf:", any(w.get('code') == 'unlikely_leaf' for w in warnings))

# Blurry
blurry = Image.new('RGB', (300, 300), (50, 120, 50)).filter(ImageFilter.GaussianBlur(radius=5))
buf_b = io.BytesIO()
blurry.save(buf_b, format='PNG')
blurry_bytes = buf_b.getvalue()
blurry_result = request(blurry_bytes, b'blurry.png')
print("BLURRY validation:", blurry_result.get('validation'))
blurry_warnings = blurry_result.get('validation', {}).get('warnings', [])
print("BLURRY warnings:", blurry_warnings)
print("BLURRY has blurry:", any(w.get('code') == 'blurry' for w in blurry_warnings))
