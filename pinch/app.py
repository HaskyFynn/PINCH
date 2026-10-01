"""Desktop controls on the main thread, camera and inference on worker threads."""
import argparse
import queue
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from .config import load_settings
from .worker import Worker


def draw_observations(image, observations):
    result = image.copy()
    for o in observations:
        x1,y1,x2,y2 = map(int, o.box)
        color = (80,200,100) if o.name != 'unknown' else (70,170,245)
        if not o.detected:
            color = (170,170,170)
            for x in range(x1,x2,12):
                cv2.line(result,(x,y1),(min(x+6,x2),y1),color,2)
                cv2.line(result,(x,y2),(min(x+6,x2),y2),color,2)
            for y in range(y1,y2,12):
                cv2.line(result,(x1,y),(x1,min(y+6,y2)),color,2)
                cv2.line(result,(x2,y),(x2,min(y+6,y2)),color,2)
        else:
            cv2.rectangle(result,(x1,y1),(x2,y2),color,2)
        label = f'T{o.track_id} {o.name}' if o.track_id >= 0 else 'New detection'
        if not o.detected:
            label += ' (predicted)'
        elif o.quality not in ('usable','enrollment'):
            label += ' / ' + o.quality.replace('_',' ')
        cv2.putText(result,label,(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
    return result


class App:
    def __init__(self, root, settings, worker=None):
        self.root, self.settings = root, settings
        self.worker = worker or Worker(settings)
        self.names = []
        self.closing = False
        root.title('PINCH — Multi-marker reader')
        root.geometry('1360x820')
        root.minsize(1080,700)
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('.', background='#18202c', foreground='#edf1f7', font=('Segoe UI',10))
        style.configure('TButton', padding=9)
        style.configure('TEntry', fieldbackground='#273447', foreground='#edf1f7')
        style.configure('Treeview', background='#1e2938', fieldbackground='#1e2938', rowheight=28)
        root.configure(bg='#18202c')
        root.columnconfigure(1,weight=1)
        root.rowconfigure(0,weight=1)
        sidebar = ttk.Frame(root)
        sidebar.grid(row=0,column=0,sticky='ns')
        scroller = tk.Canvas(sidebar,width=320,bg='#18202c',highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar,orient='vertical',command=scroller.yview)
        scroller.configure(yscrollcommand=scrollbar.set)
        scroller.pack(side='left',fill='y',expand=True)
        scrollbar.pack(side='right',fill='y')
        left = ttk.Frame(scroller,padding=18,width=320)
        scroller.create_window((0,0),window=left,anchor='nw',width=320)
        left.bind('<Configure>',lambda event:scroller.configure(scrollregion=scroller.bbox('all')))
        right = ttk.Frame(root,padding=12)
        right.grid(row=0,column=1,sticky='nsew')
        right.columnconfigure(0,weight=1)
        right.rowconfigure(0,weight=1)
        ttk.Label(left,text='PINCH',font=('Segoe UI',25,'bold')).pack(anchor='w')
        ttk.Label(left,text='Enroll once. Follow every marker.').pack(anchor='w',pady=(0,20))
        self.controls = []
        self.camera = tk.StringVar(value='0')
        camera_row = ttk.Frame(left)
        camera_row.pack(fill='x')
        ttk.Label(camera_row,text='Camera').pack(side='left')
        ttk.Spinbox(camera_row,from_=0,to=9,textvariable=self.camera,width=4).pack(side='right')
        self.button(left,'Open camera',lambda:self.open_camera())
        self.button(left,'Open video…',self.open_video)
        ttk.Separator(left).pack(fill='x',pady=14)
        self.record = tk.BooleanVar(value=False)
        ttk.Checkbutton(left,text='Save video with session logs',variable=self.record).pack(anchor='w',pady=4)
        self.button(left,'Start recognition',lambda:self.worker.send('run',record_video=self.record.get()))
        self.button(left,'Stop / cancel enrollment',lambda:self.worker.send('stop'))
        ttk.Separator(left).pack(fill='x',pady=14)
        ttk.Label(left,text='Marker name (one physical tag)').pack(anchor='w')
        self.name = tk.StringVar()
        ttk.Entry(left,textvariable=self.name).pack(fill='x',pady=6)
        self.button(left,'Enroll / replace marker',self.enroll)
        self.next_button = self.button(left,'Next view',lambda:self.worker.send('next_view'))
        self.save_button = self.button(left,'Save enrollment',lambda:self.worker.send('save_enrollment'))
        self.enrollment_text = tk.StringVar(value='Keep only the enrollment target in view.\nUse a unique name for each physical marker.')
        ttk.Label(left,textvariable=self.enrollment_text,wraplength=280).pack(fill='x',pady=12)
        self.button(left,'Import registry…',self.import_registry)
        self.button(left,'Reload saved registry',lambda:self.worker.send('reload'))
        self.registry_text = tk.StringVar(value='Loading saved markers…')
        ttk.Label(left,textvariable=self.registry_text,wraplength=280).pack(fill='x',pady=10)
        self.video = tk.Label(right,bg='#101620',fg='#9caec4',text='Loading models…',font=('Segoe UI',18))
        self.video.grid(row=0,column=0,sticky='nsew')
        self.metrics = tk.StringVar(value='Measured detections use solid boxes. Briefly predicted ROIs use dashed boxes.')
        ttk.Label(right,textvariable=self.metrics,wraplength=950).grid(row=1,column=0,sticky='ew',pady=8)
        columns = ('track','name','quality','identity','score','candidate')
        self.table = ttk.Treeview(right,columns=columns,show='headings',height=6)
        for key,title,width in zip(columns,('Track','Marker','Image quality','Identity status','Best / required','Best candidate'),(55,110,120,160,125,120)):
            self.table.heading(key,text=title)
            self.table.column(key,width=width,anchor='w')
        self.table.grid(row=2,column=0,sticky='ew')
        self.status = tk.StringVar(value='Loading models. Camera remains closed until you open it.')
        ttk.Label(root,textvariable=self.status,padding=10,wraplength=1300).grid(row=1,column=0,columnspan=2,sticky='ew')
        for b in self.controls:
            b.configure(state='disabled')
        root.protocol('WM_DELETE_WINDOW',self.close)
        self.worker.start()
        self.root.after(30,self.poll)

    def button(self,parent,text,command):
        b = ttk.Button(parent,text=text,command=command)
        b.pack(fill='x',pady=3)
        self.controls.append(b)
        return b

    def open_camera(self):
        try:
            index = int(self.camera.get())
            if not 0 <= index <= 9:
                raise ValueError()
            self.worker.send('source',kind='camera',value=index)
        except ValueError:
            messagebox.showerror('Camera','Choose a camera index from 0 to 9.',parent=self.root)

    def open_video(self):
        path = filedialog.askopenfilename(parent=self.root,title='Open recording',filetypes=[('Video','*.mp4 *.mov *.avi *.mkv'),('All files','*.*')])
        if path:
            self.worker.send('source',kind='video',value=path)

    def import_registry(self):
        path = filedialog.askopenfilename(parent=self.root,title='Import marker registry',filetypes=[('Registry','*.json')])
        if path and messagebox.askyesno('Import registry','Use this registry in place of the current one? A backup of the current saved registry will be kept.',parent=self.root):
            self.worker.send('import',path=path)

    def enroll(self):
        name = self.name.get().strip()
        if not name or name.lower() == 'unknown' or len(name)>64:
            messagebox.showerror('Marker name','Enter a unique name of 1–64 characters other than "unknown".',parent=self.root)
            return
        self.worker.send('enroll',name=name)

    def on_event(self,event):
        kind = event['event']
        if kind == 'ready':
            for b in self.controls:
                b.configure(state='normal')
            self.next_button.configure(state='disabled')
            self.save_button.configure(state='disabled')
            self.status.set('Ready on '+event['device']+'. Open a camera or video, then start recognition or enroll a tag.')
            self.video.configure(text='Open a camera or recorded video')
        elif kind == 'registry':
            self.names = event['names']
            self.registry_text.set('Saved markers: '+(', '.join(self.names) or 'none'))
            if event['warnings']:
                self.status.set('Legacy profiles loaded. Verify each marker live; re-enroll profiles made with a different model.')
        elif kind == 'status':
            self.status.set(event['message'])
        elif kind == 'mode':
            self.status.set({'preview':'Preview — choose enrollment or recognition.', 'run':'Recognition running. Each name belongs to one physical marker.',
                             'enroll':'Enrollment — only the target marker should be visible.'}[event['mode']])
            self.next_button.configure(state='disabled')
            self.save_button.configure(state='disabled')
            if event['mode'] != 'enroll':
                self.enrollment_text.set('Keep only the enrollment target in view.\nUse a unique name for each physical marker.')
        elif kind == 'enrollment':
            self.enrollment_text.set(f"View {event['step']+1}/5: {event['view']}\n{event['count']}/{event['required']} clear samples\n{event['message']}")
            self.next_button.configure(state='normal' if event['count']>=event['required'] and event['step']<4 else 'disabled')
            self.save_button.configure(state='normal' if event['complete'] else 'disabled')
        elif kind == 'confirm_profile':
            text = f"Save {event['name']} from the five collected views?"
            if event['replace']:
                text += '\nThis replaces the saved profile with the same name; a registry backup is kept.'
            if event['conflicts']:
                text += '\n\nThis marker looks similar to: '+', '.join(event['conflicts'])+'. It may remain unknown when those markers are present. Prefer a more distinct tag or enroll more representative views.'
            if messagebox.askyesno('Save enrollment',text,parent=self.root):
                self.worker.send('commit_profile')
        elif kind in ('error','fatal'):
            self.status.set(event['message'])
            messagebox.showerror('PINCH',event['message'],parent=self.root)

    def poll(self):
        if self.closing:
            if self.worker.is_alive():
                self.root.after(100,self.poll)
            else:
                self.root.destroy()
            return
        try:
            while True:
                self.on_event(self.worker.events.get_nowait())
        except queue.Empty:
            pass
        try:
            frame, observations, metrics, dropped = self.worker.frames.get_nowait()
            image = draw_observations(frame.image,observations)
            rgb = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            rgb.thumbnail((max(640,self.video.winfo_width()),max(360,self.video.winfo_height())))
            self.photo = ImageTk.PhotoImage(rgb)
            self.video.configure(image=self.photo,text='')
            self.table.delete(*self.table.get_children())
            for o in observations:
                self.table.insert('', 'end', values=(o.track_id if o.track_id>=0 else 'new',o.name,
                                  o.quality.replace('_',' '),o.identity_status.replace('_',' '),f'{o.score:.3f} / {o.threshold:.3f}' if o.score>=0 else '—',o.best_candidate))
            if metrics:
                age = (time.monotonic()-frame.captured_at)*1000
                self.metrics.set(f"Detected {metrics['raw_detections']}  |  Tracked {metrics['tracked_detections']}  |  Named {metrics['identified']}  |  Usable crops {metrics['usable_crops']}  |  Processing {metrics['processing_ms']:.0f} ms  |  Read to display {age:.0f} ms  |  Capture frames skipped {dropped}")
                if metrics['processing_ms'] > 500:
                    self.metrics.set(self.metrics.get()+'\nLow processing rate: fast movement may be missed. Test slowly and inspect the saved recording.')
        except queue.Empty:
            pass
        self.root.after(30,self.poll)

    def close(self):
        self.closing = True
        self.status.set('Closing capture and saving session logs…')
        for b in self.controls:
            b.configure(state='disabled')
        self.worker.stopping.set()


def main():
    parser = argparse.ArgumentParser(description='PINCH multi-marker desktop reader')
    parser.add_argument('--config')
    args = parser.parse_args()
    settings = load_settings(args.config)
    root = tk.Tk()
    App(root,settings)
    root.mainloop()


if __name__ == '__main__':
    main()
