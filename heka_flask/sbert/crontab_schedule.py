from crontab import CronTab
import os
path = os.getcwd()
command_str = f'python {path}/comp_emb_physio.py'
print(command_str)
cron = CronTab(user=True)
cron.remove_all()
job = cron.new(command=command_str)
job.clear()
job.minute.on(59)
job.hour.on(23)
cron.write()
