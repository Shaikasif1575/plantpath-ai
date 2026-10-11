import sqlite3, datetime, os, shutil

db = os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\User Data\Default\History")
tmp = os.path.join(os.environ["TEMP"], "chrome_hist2.db")
shutil.copyfile(db, tmp)
con = sqlite3.connect(tmp)
cur = con.cursor()
cur.execute(
    "SELECT url, title, last_visit_time FROM urls WHERE (url LIKE '%colab%' OR url LIKE '%kaggle%' OR url LIKE '%plant%' OR title LIKE '%plant%' OR title LIKE '%Colab%') ORDER BY last_visit_time DESC LIMIT 60"
)
for u, t, lv in cur.fetchall():
    when = datetime.datetime(1601, 1, 1) + datetime.timedelta(microseconds=lv / 10) if lv else None
    print(when, "|", (t or "")[:80], "|", u[:160])
con.close()
