"""GUI interaction smoke checks without a camera or model inference."""
import queue
import threading
import unittest
from unittest.mock import patch
import tkinter as tk
import numpy as np
from pinch.app import App
from pinch.config import Settings
from pinch.source import Frame
from pinch.pipeline import Observation


class FakeWorker:
    def __init__(self):
        self.events=queue.Queue()
        self.frames=queue.Queue()
        self.stopping=threading.Event()
        self.commands=[]

    def start(self):
        self.events.put({'event':'ready','device':'test'})

    def send(self,command,**args):
        self.commands.append((command,args))

    def is_alive(self):
        return False


class DesktopTests(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk()
        self.root.withdraw()
        self.worker=FakeWorker()
        self.app=App(self.root,Settings(),self.worker)
        self.app.on_event({'event':'ready','device':'test'})

    def tearDown(self):
        # Cancel scheduled poll callbacks before destroying their Tcl interpreter.
        for after_id in self.root.tk.call('after','info'):
            self.root.after_cancel(after_id)
        self.root.destroy()

    def test_source_and_enrollment_commands(self):
        self.app.open_camera()
        self.app.name.set('Marker 4')
        self.app.enroll()
        self.assertEqual(self.worker.commands,[('source',{'kind':'camera','value':0}),('enroll',{'name':'Marker 4'})])

    def test_view_progress_and_save_controls(self):
        self.app.on_event({'event':'enrollment','step':0,'view':'Front','count':8,'required':8,'total':8,'complete':False,'message':'Next'})
        self.assertEqual(str(self.app.next_button['state']),'normal')
        self.assertEqual(str(self.app.save_button['state']),'disabled')
        self.app.on_event({'event':'enrollment','step':4,'view':'Tilt','count':8,'required':8,'total':40,'complete':True,'message':'Save'})
        self.assertEqual(str(self.app.save_button['state']),'normal')

    def test_registry_and_four_rows_render(self):
        self.app.on_event({'event':'registry','names':['A','B','C','D'],'warnings':[]})
        frame=Frame(np.zeros((180,320,3),dtype=np.uint8),0,0,0)
        observations=[Observation(i,[i*60,30,i*60+50,100],.9,name=chr(65+i)) for i in range(4)]
        self.worker.frames.put((frame,observations,{},0))
        self.app.poll()
        self.root.update_idletasks()
        self.assertEqual(len(self.app.table.get_children()),4)
        self.assertIn('A, B, C, D',self.app.registry_text.get())

    def test_confirm_replacement_is_explicit(self):
        with patch('pinch.app.messagebox.askyesno',return_value=False):
            self.app.on_event({'event':'confirm_profile','name':'A','replace':True,'conflicts':['B']})
        self.assertEqual(self.worker.commands,[])
        with patch('pinch.app.messagebox.askyesno',return_value=True):
            self.app.on_event({'event':'confirm_profile','name':'A','replace':True,'conflicts':[]})
        self.assertEqual(self.worker.commands,[('commit_profile',{})])


if __name__=='__main__':
    unittest.main()
