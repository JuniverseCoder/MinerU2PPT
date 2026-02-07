import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import queue
import sys
import os
import shutil
import webbrowser
import locale
from tkinterdnd2 import DND_FILES, TkinterDnD
from converter.generator import convert_mineru_to_ppt

# --- i18n Setup ---
TRANSLATIONS = {
    "en": {
        "app_title": "File to PPT Converter",
        "input_file_label": "Input File (PDF/Image, Drag & Drop):",
        "json_file_label": "MinerU JSON File (Drag & Drop here):",
        "output_file_label": "Output PPTX File:",
        "browse_button": "Browse...",
        "save_as_button": "Save As...",
        "help_button": "?",
        "remove_watermark_checkbox": "Remove Watermark",
        "debug_images_checkbox": "Generate Debug Images",
        "start_button": "Start Conversion",
        "converting_button": "Converting...",
        "output_folder_button": "Open Output Folder",
        "debug_folder_button": "Open Debug Folder",
        "log_label": "Log",
        "json_help_title": "MinerU JSON Help",
        "json_help_text": "This tool requires a JSON file from the MinerU PDF/Image Extractor for all conversions.\n\nClick OK to open the extractor website.",
        "error_title": "Error", "info_title": "Info", "complete_title": "Complete",
        "error_all_paths": "Please fill in all file paths.",
        "error_dir_not_found": "Output directory not found: {}",
        "info_no_output": "No output file has been generated yet.",
        "info_debug_not_found": "Debug folder 'tmp' not found. Run a conversion with 'Generate Debug Images' enabled to create it.",
        "log_success": "\n--- CONVERSION FINISHED SUCCESSFULLY ---\n",
        "log_error": "\n--- ERROR ---\n{}\n",
        "msg_conversion_complete": "Conversion process has finished. Check the log for details.",
        "batch_mode_button": "Batch Mode",
        "single_mode_button": "Single Mode",
        "add_task_button": "Add Task",
        "delete_task_button": "Delete Task",
        "start_batch_button": "Start Batch Conversion",
        "task_list_label": "Task List",
        "error_no_tasks": "Please add at least one task to the list.",
        "log_batch_start": "\n--- STARTING BATCH CONVERSION ---\n",
        "log_batch_complete": "\n--- BATCH CONVERSION FINISHED ---\n",
        "log_task_start": "Starting task {} of {}: {}",
        "log_task_complete": "Finished task: {}\n",
        "add_task_title": "Add New Task",
        "ok_button": "OK",
        "cancel_button": "Cancel",
    },
    "zh": {
        "app_title": "MinerU 转 PPT 转换器",
        "input_file_label": "输入文件 (PDF/图片, 可拖拽):",
        "json_file_label": "MinerU JSON 文件 (可拖拽):",
        "output_file_label": "输出 PPTX 文件:",
        "browse_button": "浏览...",
        "save_as_button": "另存为...",
        "help_button": "？",
        "remove_watermark_checkbox": "移除水印",
        "debug_images_checkbox": "生成调试图片",
        "start_button": "开始转换",
        "converting_button": "转换中...",
        "output_folder_button": "打开输出文件夹",
        "debug_folder_button": "打开调试文件夹",
        "log_label": "日志",
        "json_help_title": "MinerU JSON 帮助",
        "json_help_text": "所有转换都需要由 MinerU PDF/图片提取器生成的 JSON 文件。\n\n点击“确定”在浏览器中打开提取器网站。",
        "error_title": "错误", "info_title": "信息", "complete_title": "完成",
        "error_all_paths": "请填写所有文件路径。",
        "error_dir_not_found": "输出目录未找到: {}",
        "info_no_output": "尚未生成输出文件。",
        "info_debug_not_found": "未找到调试文件夹 'tmp'。请在启用“生成调试图片”的情况下运行转换以创建它。",
        "log_success": "\n--- 转换成功 ---\n",
        "log_error": "\n--- 错误 ---\n{}\n",
        "msg_conversion_complete": "转换过程已结束。请查看日志了解详情。",
        "batch_mode_button": "批量模式",
        "single_mode_button": "单个模式",
        "add_task_button": "添加任务",
        "delete_task_button": "删除任务",
        "start_batch_button": "开始批量转换",
        "task_list_label": "任务列表",
        "error_no_tasks": "请至少添加一个任务到列表。",
        "log_batch_start": "\n--- 开始批量转换 ---\n",
        "log_batch_complete": "\n--- 批量转换完成 ---\n",
        "log_task_start": "开始任务 {} of {}: {}",
        "log_task_complete": "完成任务: {}\n",
        "add_task_title": "添加新任务",
        "ok_button": "确定",
        "cancel_button": "取消",
    }
}

def get_language():
    try:
        lang_code, _ = locale.getdefaultlocale()
        return 'zh' if lang_code and lang_code.lower().startswith('zh') else 'en'
    except Exception: return 'en'

class QueueHandler:
    def __init__(self, queue): self.queue = queue
    def write(self, text): self.queue.put(text)
    def flush(self): pass

class AddTaskDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.i18n = parent.i18n
        self.title(self.i18n['add_task_title'])
        self.geometry("600x220")

        self.input_path = tk.StringVar()
        self.json_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.remove_watermark = tk.BooleanVar(value=True)
        self.result = None

        self._create_widgets()
        self.transient(parent)
        self.grab_set()
        self.wait_window(self)

    def _create_widgets(self):
        frame = tk.Frame(self, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.grid_columnconfigure(1, weight=1)

        tk.Label(frame, text=self.i18n['input_file_label']).grid(row=0, column=0, sticky="w", pady=5)
        input_entry = tk.Entry(frame, textvariable=self.input_path)
        input_entry.grid(row=0, column=1, sticky="ew", padx=5)
        input_entry.drop_target_register(DND_FILES)
        input_entry.dnd_bind('<<Drop>>', lambda e: self._on_drop(e, self.input_path))
        tk.Button(frame, text=self.i18n['browse_button'], command=self._browse_input).grid(row=0, column=2, padx=5)

        tk.Label(frame, text=self.i18n['json_file_label']).grid(row=1, column=0, sticky="w", pady=5)
        json_entry = tk.Entry(frame, textvariable=self.json_path)
        json_entry.grid(row=1, column=1, sticky="ew", padx=5)
        json_entry.drop_target_register(DND_FILES)
        json_entry.dnd_bind('<<Drop>>', lambda e: self._on_drop(e, self.json_path))
        tk.Button(frame, text=self.i18n['browse_button'], command=self._browse_json).grid(row=1, column=2, padx=5)

        tk.Label(frame, text=self.i18n['output_file_label']).grid(row=2, column=0, sticky="w", pady=5)
        tk.Entry(frame, textvariable=self.output_path).grid(row=2, column=1, sticky="ew", padx=5)
        tk.Button(frame, text=self.i18n['save_as_button'], command=self._save_pptx).grid(row=2, column=2, padx=5)

        options_frame = tk.Frame(frame)
        options_frame.grid(row=3, column=0, columnspan=3, pady=10, sticky="w")
        tk.Checkbutton(options_frame, text=self.i18n['remove_watermark_checkbox'], variable=self.remove_watermark).pack(side=tk.LEFT)

        buttons_frame = tk.Frame(frame)
        buttons_frame.grid(row=4, column=0, columnspan=3, pady=5)
        tk.Button(buttons_frame, text=self.i18n['ok_button'], command=self._on_ok, width=10).pack(side=tk.LEFT, padx=10)
        tk.Button(buttons_frame, text=self.i18n['cancel_button'], command=self.destroy, width=10).pack(side=tk.LEFT, padx=10)

    def _on_drop(self, event, var):
        filepath = event.data.strip('{}')
        var.set(filepath)
        if var == self.input_path:
            self._set_default_output_path(filepath)

    def _set_default_output_path(self, in_path):
        if not self.output_path.get() and in_path:
            self.output_path.set(os.path.splitext(in_path)[0] + ".pptx")

    def _browse_input(self):
        filetypes = [("Supported Files", "*.pdf *.png *.jpg *.jpeg *.bmp"), ("All Files", "*.*")]
        path = filedialog.askopenfilename(filetypes=filetypes, parent=self)
        if path: self.input_path.set(path); self._set_default_output_path(path)

    def _browse_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")], parent=self)
        if path: self.json_path.set(path)

    def _save_pptx(self):
        path = filedialog.asksaveasfilename(defaultextension=".pptx", filetypes=[("PowerPoint Files", "*.pptx"), ("All Files", "*.*")], parent=self)
        if path: self.output_path.set(path)

    def _on_ok(self):
        input_f, json_f, output_f = self.input_path.get(), self.json_path.get(), self.output_path.get()
        if not all([input_f, json_f, output_f]):
            messagebox.showerror(self.i18n['error_title'], self.i18n['error_all_paths'], parent=self)
            return
        self.result = {"input": input_f, "json": json_f, "output": output_f, "remove_watermark": self.remove_watermark.get()}
        self.destroy()

class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.i18n = TRANSLATIONS[get_language()]
        self.title(self.i18n['app_title'])
        self.geometry("700x600")
        self.debug_folder_path = os.path.join(os.getcwd(), "tmp")
        self.input_path, self.json_path, self.output_path = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.remove_watermark, self.generate_debug = tk.BooleanVar(value=True), tk.BooleanVar(value=False)
        self.batch_mode = tk.BooleanVar(value=False)
        self.task_list = []
        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        self._create_widgets()
        self._poll_log_queue()

    def _create_widgets(self):
        self.main_frame = tk.Frame(self, padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(4, weight=1)

        self.mode_switch_button = tk.Button(self.main_frame, text=self.i18n['batch_mode_button'], command=self._toggle_batch_mode)
        self.mode_switch_button.grid(row=0, column=2, sticky="e", pady=(0, 5))

        # --- Single Mode Frame ---
        self.single_mode_frame = tk.Frame(self.main_frame)
        self.single_mode_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.single_mode_frame.grid_columnconfigure(1, weight=1)
        # (Content of single mode frame...)
        tk.Label(self.single_mode_frame, text=self.i18n['input_file_label']).grid(row=0, column=0, sticky="w", pady=2)
        input_entry = tk.Entry(self.single_mode_frame, textvariable=self.input_path, state="readonly")
        input_entry.grid(row=0, column=1, sticky="ew", padx=5)
        input_entry.drop_target_register(DND_FILES); input_entry.dnd_bind('<<Drop>>', lambda e: self._on_drop(e, self.input_path))
        tk.Button(self.single_mode_frame, text=self.i18n['browse_button'], command=self._browse_input).grid(row=0, column=2, sticky="w")

        tk.Label(self.single_mode_frame, text=self.i18n['json_file_label']).grid(row=1, column=0, sticky="w", pady=2)
        json_entry = tk.Entry(self.single_mode_frame, textvariable=self.json_path, state="readonly")
        json_entry.grid(row=1, column=1, sticky="ew", padx=5)
        json_entry.drop_target_register(DND_FILES); json_entry.dnd_bind('<<Drop>>', lambda e: self._on_drop(e, self.json_path))
        json_buttons_frame = tk.Frame(self.single_mode_frame)
        json_buttons_frame.grid(row=1, column=2, sticky="w")
        tk.Button(json_buttons_frame, text=self.i18n['browse_button'], command=self._browse_json).pack(side=tk.LEFT)
        tk.Button(json_buttons_frame, text=self.i18n['help_button'], command=self._show_json_help, width=2).pack(side=tk.LEFT)

        tk.Label(self.single_mode_frame, text=self.i18n['output_file_label']).grid(row=2, column=0, sticky="w", pady=2)
        tk.Entry(self.single_mode_frame, textvariable=self.output_path).grid(row=2, column=1, sticky="ew", padx=5)
        tk.Button(self.single_mode_frame, text=self.i18n['save_as_button'], command=self._save_pptx).grid(row=2, column=2, sticky="w")

        # --- Batch Mode Frame ---
        self.batch_frame = tk.Frame(self.main_frame)
        self.batch_frame.grid_columnconfigure(0, weight=1); self.batch_frame.grid_rowconfigure(0, weight=1)
        # (Content of batch mode frame...)
        task_list_frame = tk.LabelFrame(self.batch_frame, text=self.i18n['task_list_label'], padx=5, pady=5)
        task_list_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=5)
        task_list_frame.grid_columnconfigure(0, weight=1); task_list_frame.grid_rowconfigure(0, weight=1)
        self.task_listbox = tk.Listbox(task_list_frame, height=8)
        self.task_listbox.grid(row=0, column=0, sticky="nsew")
        task_scrollbar = tk.Scrollbar(task_list_frame, orient="vertical", command=self.task_listbox.yview)
        task_scrollbar.grid(row=0, column=1, sticky="ns"); self.task_listbox.config(yscrollcommand=task_scrollbar.set)
        batch_buttons_frame = tk.Frame(self.batch_frame)
        batch_buttons_frame.grid(row=1, column=0, columnspan=2, pady=(5,0))
        tk.Button(batch_buttons_frame, text=self.i18n['add_task_button'], command=self._add_task).pack(side=tk.LEFT, padx=5)
        tk.Button(batch_buttons_frame, text=self.i18n['delete_task_button'], command=self._delete_task).pack(side=tk.LEFT, padx=5)

        # --- Options (will be managed dynamically) ---
        self.options_frame = tk.Frame(self.main_frame)
        self.options_frame.grid(row=2, column=0, columnspan=3, pady=10, sticky="w")
        self.remove_watermark_checkbox = tk.Checkbutton(self.options_frame, text=self.i18n['remove_watermark_checkbox'], variable=self.remove_watermark)
        self.remove_watermark_checkbox.pack(side=tk.LEFT, padx=5)
        self.debug_images_checkbox = tk.Checkbutton(self.options_frame, text=self.i18n['debug_images_checkbox'], variable=self.generate_debug, command=self._toggle_debug_button_visibility)
        self.debug_images_checkbox.pack(side=tk.LEFT, padx=5)

        # --- Actions and Log ---
        action_frame = tk.Frame(self.main_frame)
        action_frame.grid(row=3, column=0, columnspan=3, pady=10)
        action_frame.grid_columnconfigure(0, weight=1)
        button_container = tk.Frame(action_frame)
        button_container.grid(row=0, column=0)
        self.start_button = tk.Button(button_container, text=self.i18n['start_button'], command=self.start_conversion_thread)
        self.start_button.pack(side=tk.LEFT, padx=10)
        self.output_button = tk.Button(button_container, text=self.i18n['output_folder_button'], command=self._open_output_folder, state="disabled")
        self.output_button.pack(side=tk.LEFT, padx=10)
        self.debug_button = tk.Button(button_container, text=self.i18n['debug_folder_button'], command=self._open_debug_folder, state="disabled")

        log_frame = tk.LabelFrame(self.main_frame, text=self.i18n['log_label'], padx=5, pady=5)
        log_frame.grid(row=4, column=0, columnspan=3, sticky="nsew")
        log_frame.grid_rowconfigure(0, weight=1); log_frame.grid_columnconfigure(0, weight=1)
        self.log_area = scrolledtext.ScrolledText(log_frame, state="disabled", wrap=tk.WORD, height=10)
        self.log_area.grid(row=0, column=0, sticky="nsew")

        self._toggle_batch_mode() # Set initial state to single mode
        self._toggle_batch_mode()

    def _toggle_batch_mode(self):
        is_batch = not self.batch_mode.get()
        self.batch_mode.set(is_batch)
        if is_batch:
            self.single_mode_frame.grid_remove()
            self.batch_frame.grid(row=1, column=0, columnspan=3, sticky="nsew")
            self.mode_switch_button.config(text=self.i18n['single_mode_button'])
            self.start_button.config(text=self.i18n['start_batch_button'])
            # Hide options not relevant to batch mode
            self.options_frame.grid_remove()
            self.debug_button.pack_forget()
        else:
            self.batch_frame.grid_remove()
            self.single_mode_frame.grid()
            self.mode_switch_button.config(text=self.i18n['batch_mode_button'])
            self.start_button.config(text=self.i18n['start_button'])
            # Show options for single mode
            self.options_frame.grid()
            self._toggle_debug_button_visibility()

    def _add_task(self):
        dialog = AddTaskDialog(self)
        if dialog.result:
            task = dialog.result
            self.task_list.append(task)
            suffix = "" if task['remove_watermark'] else " (Keep WM)"
            self.task_listbox.insert(tk.END, f"IN: {os.path.basename(task['input'])} -> OUT: {os.path.basename(task['output'])}{suffix}")

    def _delete_task(self):
        selected_indices = self.task_listbox.curselection()
        if not selected_indices: return
        for index in sorted(selected_indices, reverse=True):
            self.task_listbox.delete(index)
            del self.task_list[index]

    def _show_json_help(self):
        if messagebox.askokcancel(self.i18n['json_help_title'], self.i18n['json_help_text']):
            webbrowser.open_new("https://mineru.net/OpenSourceTools/Extractor")

    def _toggle_debug_button_visibility(self):
        if self.generate_debug.get() and not self.batch_mode.get():
            self.debug_button.pack(side=tk.LEFT, padx=10)
        else:
            self.debug_button.pack_forget()

    def _open_output_folder(self):
        output_file = self.output_path.get()
        if self.batch_mode.get() and self.task_list:
             output_file = self.task_list[-1]['output']
        if not output_file:
            messagebox.showinfo(self.i18n['info_title'], self.i18n['info_no_output']); return
        output_dir = os.path.dirname(output_file)
        if os.path.exists(output_dir): os.startfile(output_dir)
        else: messagebox.showerror(self.i18n['error_title'], self.i18n['error_dir_not_found'].format(output_dir))

    def _open_debug_folder(self):
        if os.path.exists(self.debug_folder_path): os.startfile(self.debug_folder_path)
        else: messagebox.showinfo(self.i18n['info_title'], self.i18n['info_debug_not_found'])

    def _set_default_output_path(self, in_path):
        if not self.output_path.get(): self.output_path.set(os.path.splitext(in_path)[0] + ".pptx")

    def _on_drop(self, event, var):
        filepath = event.data.strip('{}')
        var.set(filepath)
        if var == self.input_path: self._set_default_output_path(filepath)

    def _browse_input(self):
        filetypes = [("Supported Files", "*.pdf *.png *.jpg *.jpeg *.bmp"), ("All Files", "*.*")]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path: self.input_path.set(path); self._set_default_output_path(path)

    def _browse_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if path: self.json_path.set(path)

    def _save_pptx(self):
        path = filedialog.asksaveasfilename(defaultextension=".pptx", filetypes=[("PowerPoint Files", "*.pptx"), ("All Files", "*.*")])
        if path: self.output_path.set(path)

    def _poll_log_queue(self):
        while True:
            try:
                record = self.log_queue.get_nowait()
                self.log_area.config(state="normal"); self.log_area.insert(tk.END, record); self.log_area.see(tk.END); self.log_area.config(state="disabled")
            except queue.Empty: break
        self.after(100, self._poll_log_queue)

    def start_conversion_thread(self):
        if self.batch_mode.get():
            if not self.task_list:
                messagebox.showerror(self.i18n['error_title'], self.i18n['error_no_tasks'])
                return
            target_func, args = self._run_batch_conversion, ()
        else:
            input_file, json_f = self.input_path.get(), self.json_path.get()
            if not self.output_path.get() and input_file: self._set_default_output_path(input_file)
            output = self.output_path.get()
            if not all([input_file, json_f, output]):
                messagebox.showerror(self.i18n['error_title'], self.i18n['error_all_paths']); return
            target_func, args = self._run_single_conversion, (json_f, input_file, output)

        self.start_button.config(state="disabled", text=self.i18n['converting_button'])
        self.output_button.config(state="disabled")
        if not self.batch_mode.get(): self.debug_button.config(state="disabled")
        self.log_area.config(state="normal"); self.log_area.delete(1.0, tk.END); self.log_area.config(state="disabled")

        threading.Thread(target=self._run_conversion_wrapper, args=(target_func, args), daemon=True).start()

    def _run_conversion_wrapper(self, conversion_func, args):
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = self.queue_handler, self.queue_handler
        success = False
        try:
            conversion_func(*args)
            success = True
        except Exception as e:
            self.log_queue.put(self.i18n['log_error'].format(e))
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            self.after(0, self._finalize_gui, success)

    def _run_single_conversion(self, json_path, input_path, output_path):
        args = (json_path, input_path, output_path, self.remove_watermark.get(), self.generate_debug.get())
        convert_mineru_to_ppt(*args)
        self.log_queue.put(self.i18n['log_success'])

    def _run_batch_conversion(self):
        self.log_queue.put(self.i18n['log_batch_start'])
        total_tasks = len(self.task_list)
        for i, task in enumerate(self.task_list):
            self.log_queue.put(self.i18n['log_task_start'].format(i + 1, total_tasks, os.path.basename(task['input'])))
            try:
                # Debug images are disabled for batch mode
                args = (task['json'], task['input'], task['output'], task['remove_watermark'], False)
                convert_mineru_to_ppt(*args)
                self.log_queue.put(self.i18n['log_task_complete'].format(os.path.basename(task['input'])))
            except Exception as e:
                self.log_queue.put(self.i18n['log_error'].format(e))
        self.log_queue.put(self.i18n['log_batch_complete'])

    def _finalize_gui(self, success):
        start_text = self.i18n['start_batch_button'] if self.batch_mode.get() else self.i18n['start_button']
        self.start_button.config(state="normal", text=start_text)
        if success:
            self.output_button.config(state="normal")
            if self.generate_debug.get() and not self.batch_mode.get():
                 self.debug_button.config(state="normal")
        messagebox.showinfo(self.i18n['complete_title'], self.i18n['msg_conversion_complete'])

if __name__ == "__main__":
    app = App()
    app.mainloop()
