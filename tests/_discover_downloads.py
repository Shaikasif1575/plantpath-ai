import sqlite3, datetime, os, shutil, glob

srcs = []
for pat in [
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\User Data\Default\History"),
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\User Data\Default\History-journal"),
]:
    if os.path.exists(pat):
        srcs.append(pat)

tmp = os.path.join(os.environ["TEMP"], "chrome_hist_copy.db")
if srcs:
    shutil.copyfile(srcs[0], tmp)
    con = sqlite3.connect(tmp)
    cur = con.cursor()
    try:
        cur.execute(
            "SELECT target_path, tab_url, start_time FROM downloads ORDER BY start_time DESC LIMIT 40"
        )
        for p, u, s in cur.fetchall():
            t = None
            if s:
                t = datetime.datetime(1601, 1, 1) + datetime.timedelta(microseconds=s / 10)
            print(t, "|", p, "|", u)
    except Exception as ex:
        print("ERR", ex)
    con.close()

# Edge too
edge = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\User Data\Default\History")
if os.path.exists(edge):
    tmp2 = os.path.join(os.environ["TEMP"], "edge_hist_copy.db")
    shutil.copyfile(edge, tmp2)
    con = sqlite3.connect(tmp2)
    cur = con.cursor()
    print("\n--- EDGE ---")
    try:
        cur.execute(
            "SELECT target_path, tab_url, start_time FROM downloads ORDER BY start_time DESC LIMIT 40"
        )
        for p, u, s in cur.fetchall():
            t = None
            if s:
                t = datetime.datetime(1601, 1, 1) + datetime.timedelta(microseconds=s / 10)
            print(t, "|", p, "|", u)
    except Exception as ex:
        print("ERR", ex)
    con.close()
