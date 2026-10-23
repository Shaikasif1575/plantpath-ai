import io
from PIL import Image, ImageFilter, ImageStat

def check(name, path_or_img, size=(300,300)):
    if isinstance(path_or_img, str):
        img = Image.open(path_or_img).convert("RGB")
    else:
        img = path_or_img
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    mean_brightness = float(stat.mean[0])
    rgb_stat = ImageStat.Stat(img)
    r_mean = float(rgb_stat.mean[0])
    g_mean = float(rgb_stat.mean[1])
    b_mean = float(rgb_stat.mean[2])
    green_ratio = g_mean / max(r_mean + b_mean, 1.0)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    edge_var = float(edge_stat.stddev[0]) if edge_stat.stddev else 0.0
    print(f"{name}: brightness={mean_brightness:.1f}, green_ratio={green_ratio:.2f}, edge_var={edge_var:.2f}, unlikely_leaf={green_ratio < 0.6 and mean_brightness > 30}, blurry={edge_var < 8.0}")

check("red image", Image.new("RGB", (300,300), (255,30,30)))
blurry = Image.new("RGB", (300,300), (50,120,50)).filter(ImageFilter.GaussianBlur(radius=5))
check("blurry leaf", blurry)
real = Image.open("tests/probe_set/author_sample_apple_scab.jpg").convert("RGB").resize((300,300))
check("real apple scab", real)
