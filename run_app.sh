gunicorn --bind 0.0.0.0:5000 --access-logfile - --access-logformat '%(h)s %(l)s %(t)s "%(r)s" %(s)s "Time: %(T)ss" "%(a)s"' app:app