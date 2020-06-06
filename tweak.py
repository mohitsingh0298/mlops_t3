#!/usr/bin/env python
# coding: utf-8

# In[11]:


line_index = 76
lines = None
with open('/root/task3.py', 'r') as file_handler:
    lines = file_handler.readlines()

lines.insert(line_index, 'CRP(8)\n')

with open('/root/task3.py', 'w') as file_handler:
    file_handler.writelines(lines)


# In[ ]:





# In[ ]:




