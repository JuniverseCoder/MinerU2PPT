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
        "msg_conversion_complete": "Conversion process has finished. Check the log for details."
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
        "msg_conversion_complete": "转换过程已结束。请查看日志了解详情。"
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

class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.i18n = TRANSLATIONS[get_language()]
        self.title(self.i18n['app_title'])
        self.geometry("700x550")
        self.debug_folder_path = os.path.join(os.getcwd(), "tmp")
        self.input_path, self.json_path, self.output_path = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.remove_watermark, self.generate_debug = tk.BooleanVar(value=True), tk.BooleanVar(value=False)
        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        self._create_widgets()
        self._poll_log_queue()

    def _create_widgets(self):
        main_frame = tk.Frame(self, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(5, weight=1)

        # --- Input File ---
        tk.Label(main_frame, text=self.i18n['input_file_label']).grid(row=0, column=0, sticky="w", pady=2)
        input_entry = tk.Entry(main_frame, textvariable=self.input_path, state="readonly")
        input_entry.grid(row=0, column=1, sticky="ew", padx=5)
        input_entry.drop_target_register(DND_FILES)
        input_entry.dnd_bind('<<Drop>>', lambda e: self._on_drop(e, self.input_path))
        tk.Button(main_frame, text=self.i18n['browse_button'], command=self._browse_input).grid(row=0, column=2, sticky="w")

        # --- JSON File ---
        tk.Label(main_frame, text=self.i18n['json_file_label']).grid(row=1, column=0, sticky="w", pady=2)
        json_entry = tk.Entry(main_frame, textvariable=self.json_path, state="readonly")
        json_entry.grid(row=1, column=1, sticky="ew", padx=5)
        json_entry.drop_target_register(DND_FILES)
        json_entry.dnd_bind('<<Drop>>', lambda e: self._on_drop(e, self.json_path))

        json_buttons_frame = tk.Frame(main_frame)
        json_buttons_frame.grid(row=1, column=2, sticky="w")
        tk.Button(json_buttons_frame, text=self.i18n['browse_button'], command=self._browse_json).pack(side=tk.LEFT)
        tk.Button(json_buttons_frame, text=self.i18n['help_button'], command=self._show_json_help, width=2).pack(side=tk.LEFT) # Removed padding

        # --- Output File ---
        tk.Label(main_frame, text=self.i18n['output_file_label']).grid(row=2, column=0, sticky="w", pady=2)
        tk.Entry(main_frame, textvariable=self.output_path).grid(row=2, column=1, sticky="ew", padx=5)
        tk.Button(main_frame, text=self.i18n['save_as_button'], command=self._save_pptx).grid(row=2, column=2, sticky="w")

        # --- Options ---
        options_frame = tk.Frame(main_frame)
        options_frame.grid(row=3, column=0, columnspan=3, pady=10, sticky="w")
        tk.Checkbutton(options_frame, text=self.i18n['remove_watermark_checkbox'], variable=self.remove_watermark).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(options_frame, text=self.i18n['debug_images_checkbox'], variable=self.generate_debug, command=self._toggle_debug_button_visibility).pack(side=tk.LEFT, padx=5)

        # --- Action Buttons ---
        action_frame = tk.Frame(main_frame)
        action_frame.grid(row=4, column=0, columnspan=3, pady=10)
        action_frame.grid_columnconfigure(0, weight=1) # Center the button container
        button_container = tk.Frame(action_frame)
        button_container.grid(row=0, column=0)
        self.start_button = tk.Button(button_container, text=self.i18n['start_button'], command=self.start_conversion_thread)
        self.start_button.pack(side=tk.LEFT, padx=10)
        self.output_button = tk.Button(button_container, text=self.i18n['output_folder_button'], command=self._open_output_folder, state="disabled")
        self.output_button.pack(side=tk.LEFT, padx=10)
        self.debug_button = tk.Button(button_container, text=self.i18n['debug_folder_button'], command=self._open_debug_folder, state="disabled")

        # --- Log Area ---
        log_frame = tk.LabelFrame(main_frame, text=self.i18n['log_label'], padx=5, pady=5)
        log_frame.grid(row=5, column=0, columnspan=3, sticky="nsew")
        log_frame.grid_rowconfigure(0, weight=1); log_frame.grid_columnconfigure(0, weight=1)
        self.log_area = scrolledtext.ScrolledText(log_frame, state="disabled", wrap=tk.WORD, height=10)
        self.log_area.grid(row=0, column=0, sticky="nsew")

    def _show_json_help(self):
        if messagebox.askokcancel(self.i18n['json_help_title'], self.i18n['json_help_text']):
            webbrowser.open_new("https://mineru.net/OpenSourceTools/Extractor")

    def _toggle_debug_button_visibility(self):
        if self.generate_debug.get(): self.debug_button.pack(side=tk.LEFT, padx=10)
        else: self.debug_button.pack_forget()

    def _open_output_folder(self):
        output_file = self.output_path.get()
        if not output_file: messagebox.showinfo(self.i18n['info_title'], self.i18n['info_no_output']); return
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
        input_file, json = self.input_path.get(), self.json_path.get()
        if not self.output_path.get() and input_file: self._set_default_output_path(input_file)
        output = self.output_path.get()
        if not all([input_file, json, output]): messagebox.showerror(self.i18n['error_title'], self.i18n['error_all_paths']); return

        self.start_button.config(state="disabled", text=self.i18n['converting_button'])
        self.debug_button.config(state="disabled"); self.output_button.config(state="disabled")
        self.log_area.config(state="normal"); self.log_area.delete(1.0, tk.END); self.log_area.config(state="disabled")

        args = (json, input_file, output, self.remove_watermark.get(), self.generate_debug.get())
        threading.Thread(target=self._run_conversion, args=(convert_mineru_to_ppt, args), daemon=True).start()

    def _run_conversion(self, target_func, args):
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = self.queue_handler, self.queue_handler
        success = False
        try:
            target_func(*args)
            self.log_queue.put(self.i18n['log_success'])
            success = True
        except Exception as e:
            self.log_queue.put(self.i18n['log_error'].format(e))
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            self.after(0, self._finalize_gui, self.generate_debug.get(), success)

    def _finalize_gui(self, debug_images, success):
        self.start_button.config(state="normal", text=self.i18n['start_button'])
        if success:
            self.output_button.config(state="normal")
            if debug_images: self.debug_button.config(state="normal")
        messagebox.showinfo(self.i18n['complete_title'], self.i18n['msg_conversion_complete'])

if __name__ == "__main__":
    from tkinterdnd2 import TkinterDnD
    app = App()
    app.mainloop()
