from tkinter import ttk, messagebox, filedialog
from tkcalendar import DateEntry
import tkinter as tk
import sqlite3
from PIL import Image, ImageTk
import customtkinter as ctk
import numpy as np
from matplotlib.figure import Figure
##################################################
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AnimatedLoginForm(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FAKE & REAL IMAGE DETECTION APPLICATION")
        self.geometry("800x600")
        self.dark_mode = False  # Track dark mode state
        ##self.my_image=self.my_image
        self.frames = {}
        self.file_path= None  # Initialize file_path
        self.real_fake_text_field = None
        self.real_fake_text_field1 = None
        
        # Create all necessary frames
        self.create_login_frame()
        self.create_register_frame()
        self.create_display_frame()
        self.create_home_frames()
        
        # Start with the login frame
        self.show_frame("LoginFrame")
        
    def create_login_frame(self):
        login_frame = ctk.CTkFrame(self, width=600, height=500)
        login_frame.pack_propagate(False)
        
        self.login_form_label = ctk.CTkLabel(login_frame, text="LOGIN FORM", font=("Algerian", 30))
        self.login_form_label.pack(pady=10)

        self.username_label = ctk.CTkLabel(login_frame, text="Username:", font=("Times New Roman", 16))
        self.username_label.pack(pady=(10, 0))

        self.username_input = ctk.CTkEntry(login_frame, width=300, height=30, font=("Times New Roman", 16))
        self.username_input.pack(pady=(0, 10))

        self.password_label = ctk.CTkLabel(login_frame, text="Password:", font=("Times New Roman", 16))
        self.password_label.pack(pady=(10, 0))

        self.password_input = ctk.CTkEntry(login_frame, width=300, height=30, show="*", font=("Times New Roman", 16))
        self.password_input.pack(pady=(0, 10))

        self.login_methods_label = ctk.CTkLabel(login_frame, text="Select Login Method:", font=("Times New Roman", 16))
        self.login_methods_label.pack(pady=(10, 0))

        self.login_method_combobox = ctk.CTkComboBox(login_frame, values=["Select Login Method", "Management", "Vendor"], width=300, height=30, state='readonly', font=("Times New Roman", 16))
        self.login_method_combobox.set("Select Login Method")
        self.login_method_combobox.pack(pady=(0, 10))

        self.login_button = ctk.CTkButton(login_frame, text="Login", height=30, command=self.login, fg_color="red")
        self.login_button.pack(pady=10)
        
        self.register_button = ctk.CTkButton(login_frame, text="Register", height=30, command=lambda: self.show_frame("RegisterFrame"), fg_color="blue")
        self.register_button.pack(pady=10)
        
        self.frames["LoginFrame"] = login_frame
    
    def create_register_frame(self):
        register_frame = ctk.CTkFrame(self, width=600, height=500)
        register_frame.pack_propagate(False)
        
        self.register_form_label = ctk.CTkLabel(register_frame, text="REGISTER FORM", font=("Algerian", 30))
        self.register_form_label.pack(pady=10)
        
        self.username_label_reg = ctk.CTkLabel(register_frame, text="Username:", font=("Times New Roman", 16))
        self.username_label_reg.pack(pady=(10, 0))

        self.username_input_reg = ctk.CTkEntry(register_frame, width=300, height=30, font=("Times New Roman", 16))
        self.username_input_reg.pack(pady=(0, 10))

        self.password_label_reg = ctk.CTkLabel(register_frame, text="Password:", font=("Times New Roman", 16))
        self.password_label_reg.pack(pady=(10, 0))

        self.password_input_reg = ctk.CTkEntry(register_frame, width=300, height=30, show="*", font=("Times New Roman", 16))
        self.password_input_reg.pack(pady=(0, 10))

        self.confirm_password_label = ctk.CTkLabel(register_frame, text="Confirm Password:", font=("Times New Roman", 16))
        self.confirm_password_label.pack(pady=(10, 0))

        self.confirm_password_input = ctk.CTkEntry(register_frame, width=300, height=30, show="*", font=("Times New Roman", 16))
        self.confirm_password_input.pack(pady=(0, 10))

        self.registration_date_label = ctk.CTkLabel(register_frame, text="Registration Date:", font=("Times New Roman", 16))
        self.registration_date_label.pack(pady=(10, 0))

        self.registration_date_picker = DateEntry(register_frame, font=("Times New Roman", 16), date_pattern="dd/MM/yyyy")
        self.registration_date_picker.pack(pady=(0, 10))

        self.login_methods_label_reg = ctk.CTkLabel(register_frame, text="Select Login Method:", font=("Times New Roman", 16))
        self.login_methods_label_reg.pack(pady=(10, 0))

        self.login_method_combobox_reg = ctk.CTkComboBox(register_frame, values=["Select Login Method", "Management", "Vendor"], width=300, height=30, state='readonly', font=("Times New Roman", 16))
        self.login_method_combobox_reg.set("Select Login Method")
        self.login_method_combobox_reg.pack(pady=(0, 10))

        self.create_account_button = ctk.CTkButton(register_frame, text="Create Account", width=300, height=30, command=self.create_account, fg_color="green")
        self.create_account_button.pack(pady=10)

        self.back_to_login_button = ctk.CTkButton(register_frame, text="Back to Login", width=300, height=30, command=lambda: self.show_frame("LoginFrame"), fg_color="blue")
        self.back_to_login_button.pack(pady=10)

        self.back_to_display_button = ctk.CTkButton(register_frame, text="View Records", width=300, height=30, command=lambda: self.show_frame("DisplayFrame"), fg_color="blue")
        self.back_to_display_button.pack(pady=10)
        
        self.frames["RegisterFrame"] = register_frame
    
    def create_display_frame(self):
        display_frame = ctk.CTkFrame(self, width=600, height=500)
        display_frame.pack_propagate(False)
        
        self.display_label = ctk.CTkLabel(display_frame, text="RECORDS", font=("Algerian", 30))
        self.display_label.pack(pady=10)

        self.tree = ttk.Treeview(display_frame, columns=("User_Name", "User_Password", "User_confirm_Password", "Registration_date", "User_Login_Methods"), show="headings")
        self.tree.heading("User_Name", text="User Name")
        self.tree.heading("User_Password", text="Password")
        self.tree.heading("User_confirm_Password", text="Confirm Password")
        self.tree.heading("Registration_date", text="Registration Date")
        self.tree.heading("User_Login_Methods", text="Login Method")
        self.tree.pack(expand=True, fill="both")
        
        self.back_button = ctk.CTkButton(display_frame, text="Back", height=30, command=lambda: self.show_frame("RegisterFrame"), fg_color="blue")
        self.back_button.pack(pady=10)

        self.refresh_button = ctk.CTkButton(display_frame, text="Refresh", height=30, command=self.display_records, fg_color="green")
        self.refresh_button.pack(pady=10)

        self.delete_button = ctk.CTkButton(display_frame, text="Delete", height=30, command=self.delete_selected_record, fg_color="red")
        self.delete_button.pack(pady=10)

        self.frames["DisplayFrame"] = display_frame
    
    def create_home_frames(self):
        # Management Home Frame with Tabs
        management_home_frame = ctk.CTkFrame(self, width=600, height=500)
        management_home_frame.pack_propagate(False)
        
        notebook_mgmt = ttk.Notebook(management_home_frame)
        notebook_mgmt.pack(expand=True, fill='both')

        tab1 = ctk.CTkFrame(notebook_mgmt)
        tab2 = ctk.CTkFrame(notebook_mgmt)
        
        notebook_mgmt.add(tab1, text="Management Dashboard")
        notebook_mgmt.add(tab2, text="View Management Model Records")
        
        top_frame = ctk.CTkFrame(tab1)
        top_frame.pack(fill='x', pady=10, padx=10)

        self.management_home_label = ctk.CTkLabel(top_frame, text="MANAGEMENT DASHBOARD", font=("Algerian", 30))
        self.management_home_label.pack(side="left", padx=(0, 20))

        self.logout = ctk.CTkButton(top_frame, text="Logout", command=self.logout1)
        self.logout.pack(side="left")

        left_right_frame_mgmt = ctk.CTkFrame(tab1)
        left_right_frame_mgmt.pack(fill="both", expand=True, padx=10, pady=10)

        left_frame_mgmt = ctk.CTkFrame(left_right_frame_mgmt, width=150)
        left_frame_mgmt.pack(side="left", fill="y", padx=5, pady=5)

        right_frame_mgmt = ctk.CTkFrame(left_right_frame_mgmt)
        right_frame_mgmt.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        self.upload_button = ctk.CTkButton(left_frame_mgmt, text="Upload", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.image_label = ctk.CTkLabel(left_frame_mgmt)
        self.image_label.pack(pady=10)

        self.file_entry = ctk.CTkEntry(left_frame_mgmt, width=150, height=30)
        self.file_entry.pack(pady=10)

        self.train_button = ctk.CTkButton(left_frame_mgmt, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)

        self.test_button = ctk.CTkButton(left_frame_mgmt, text="Test Model", command=self.test_model)
        self.test_button.pack(pady=10)

        self.real_label = ctk.CTkLabel(left_frame_mgmt, text="Image Status", font=("Algerian", 16))
        self.real_label.pack(pady=10)

        self.fake_real_field = ctk.CTkEntry(left_frame_mgmt, width=150, height=30)
        self.fake_real_field.pack(pady=10)

        self.submit_record = ctk.CTkButton(left_frame_mgmt, text="Submit Records", command=self.save_record)
        self.submit_record.pack(pady=10)

        # Add Scrollable Text area on the right frame
        text_frame_mgmt = ctk.CTkFrame(right_frame_mgmt)
        text_frame_mgmt.pack(fill="both", expand=True)
        #==========================================================================================================
        scrollbar_mgmt_y = tk.Scrollbar(text_frame_mgmt, orient=tk.VERTICAL)
        scrollbar_mgmt_y.pack(side="right", fill="y")
        scrollbar_mgmt_x = tk.Scrollbar(text_frame_mgmt, orient=tk.HORIZONTAL)
        scrollbar_mgmt_x.pack(side="bottom", fill="x")
        ###########################################################################################
        self.text_area = tk.Text(text_frame_mgmt, wrap=tk.WORD,yscrollcommand=scrollbar_mgmt_y.set, xscrollcommand=scrollbar_mgmt_x.set)
        self.text_area.pack(side="left", fill="both", expand=True)
        ##########################################################################################################
        scrollbar_mgmt_y.config(command=self.text_area.yview)
        scrollbar_mgmt_x.config(command=self.text_area.xview)
        #################################################################################
        tab2_frame = ctk.CTkFrame(tab2)
        tab2_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.mgmt_tree = ttk.Treeview(tab2_frame, columns=("ID", "Uploaded_Image", "Detection_Status"), show="headings")
        self.mgmt_tree.heading("ID", text="ID")
        self.mgmt_tree.heading("Uploaded_Image", text="Uploaded Image")
        self.mgmt_tree.heading("Detection_Status", text="Detection Status")
        self.mgmt_tree.pack(expand=True, fill="both")

        refresh_button_mgmt = ctk.CTkButton(tab2_frame, text="Refresh Records", height=30, fg_color="green",command=self.refresh_mgmt_records)
        refresh_button_mgmt.pack(pady=10)
        ##self.text_area.config(yscrollcommand=scrollbar_mgmt.set)
        ##command=self.refresh_mgmt_records,
        self.frames["ManagementHomeFrame"] = management_home_frame

        # Vendor Home Frame with Tabs
        vendor_home_frame = ctk.CTkFrame(self, width=600, height=500)
        vendor_home_frame.pack_propagate(False)
        
        notebook_vend = ttk.Notebook(vendor_home_frame)
        notebook_vend.pack(expand=True, fill='both')

        tab3 = ctk.CTkFrame(notebook_vend)
        tab4 = ctk.CTkFrame(notebook_vend)
        
        notebook_vend.add(tab3, text="Vendor Dashboard")
        notebook_vend.add(tab4, text="View Vendor Model Records")

        top_frame1 = ctk.CTkFrame(tab3)
        top_frame1.pack(fill='x', pady=10, padx=10)

        self.vendor_home_label = ctk.CTkLabel(top_frame1, text="VENDOR DASHBOARD", font=("Algerian", 30))
        self.vendor_home_label.pack(side="left", padx=(0, 20))

        self.logut_outbtn = ctk.CTkButton(top_frame1, text="Logout", command=self.logout_btn)
        self.logut_outbtn.pack(side="left")

        left_right_frame_vend = ctk.CTkFrame(tab3)
        left_right_frame_vend.pack(fill="both", expand=True, padx=10, pady=10)

        left_frame_vend = ctk.CTkFrame(left_right_frame_vend, width=150)
        left_frame_vend.pack(side="left", fill="y", padx=5, pady=5)

        right_frame_vend = ctk.CTkFrame(left_right_frame_vend)
        right_frame_vend.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        self.upload_button_vend = ctk.CTkButton(left_frame_vend, text="Upload", command=self.upload_image1)
        self.upload_button_vend.pack(pady=10)

        self.image_label_vend = ctk.CTkLabel(left_frame_vend)
        self.image_label_vend.pack(pady=10)

        self.file_entry_vend = ctk.CTkEntry(left_frame_vend, width=150, height=30)
        self.file_entry_vend.pack(pady=10)

        self.train_button_vend = ctk.CTkButton(left_frame_vend, text="Train Model", command=self.train_model1)
        self.train_button_vend.pack(pady=10)

        self.test_button_vend = ctk.CTkButton(left_frame_vend, text="Test Model", command=self.test_model1)
        self.test_button_vend.pack(pady=10)

        self.real_label1 = ctk.CTkLabel(left_frame_vend, text="Image Status", font=("Algerian", 16))
        self.real_label1.pack(pady=10)

        self.fake_real_field1 = ctk.CTkEntry(left_frame_vend, width=150, height=30)
        self.fake_real_field1.pack(pady=10)

        self.submit_record1 = ctk.CTkButton(left_frame_vend, text="Submit Records", command=self.save_record1)
        self.submit_record1.pack(pady=10)

        # Add Scrollable Text area on the right frame
        text_frame_vend = ctk.CTkFrame(right_frame_vend)
        text_frame_vend.pack(fill="both", expand=True)

        self.text_area_vend = tk.Text(text_frame_vend, wrap=tk.WORD)
        self.text_area_vend.pack(side="left", fill="both", expand=True)

        scrollbar_vend = tk.Scrollbar(text_frame_vend, command=self.text_area_vend.yview)
        scrollbar_vend.pack(side="right", fill="y")
        ######################################################################################

        self.text_area_vend.config(yscrollcommand=scrollbar_vend.set)
        #######################################################################################################################
        vendor_home_frame1= ctk.CTkFrame(tab4)
        vendor_home_frame1.pack(fill='both', expand=True, padx=10, pady=10)
        self.vendor_tree = ttk.Treeview(vendor_home_frame1, columns=("ID", "Uploaded_Image", "Detection_Status"), show="headings")
        self.vendor_tree.heading("ID", text="ID")
        self.vendor_tree.heading("Uploaded_Image", text="Uploaded Image")
        self.vendor_tree.heading("Detection_Status", text="Detection Status")
        self.vendor_tree.pack(expand=True, fill="both")
        refresh_button_vendor = ctk.CTkButton(vendor_home_frame1, text="Refresh Records", height=30, fg_color="green",command=self.refresh_vendor_records)
        refresh_button_vendor.pack(pady=10)
        #########################################################

        self.frames["VendorHomeFrame"] = vendor_home_frame


    def show_frame(self, frame_name):
        frame = self.frames.get(frame_name)
        if frame:
            for f in self.frames.values():
                f.pack_forget()
            frame.pack(expand=True, fill="both")

    def login(self):
        username = self.username_input.get()
        password = self.password_input.get()
        login_method = self.login_method_combobox.get()

        conn = sqlite3.connect("ML_DB.db")
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM Registration 
            WHERE User_Name=? AND User_Password=? AND User_Login_Methods=?
        """, (username, password, login_method))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            if login_method == "Management":
                self.show_frame("ManagementHomeFrame")
            elif login_method == "Vendor":
                self.show_frame("VendorHomeFrame")
        else:
            messagebox.showerror("Error", "Invalid login credentials!")

    def create_account(self):
        username = self.username_input_reg.get()
        password = self.password_input_reg.get()
        confirm_password = self.confirm_password_input.get()
        registration_date = self.registration_date_picker.get()
        login_method = self.login_method_combobox_reg.get()
        
        if password != confirm_password:
            messagebox.showerror("Error", "Passwords do not match!")
            return
        
        conn = sqlite3.connect("ML_DB.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Registration (
                User_Name TEXT,
                User_Password TEXT,
                User_confirm_Password TEXT,
                Registration_date TEXT,
                User_Login_Methods TEXT
            )
        """)
        
        cursor.execute("INSERT INTO Registration VALUES (?, ?, ?, ?, ?)", (username, password, confirm_password, registration_date, login_method))
        conn.commit()
        conn.close()
        
        messagebox.showinfo("Success", "Account created successfully!")
        self.show_frame("LoginFrame")

    def display_records(self):
        conn = sqlite3.connect("ML_DB.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Registration")
        rows = cursor.fetchall()
        conn.close()
        
        for row in self.tree.get_children():
            self.tree.delete(row)
        
        for row in rows:
            self.tree.insert("", "end", values=row)
    def refresh_mgmt_records(self):
        conn = sqlite3.connect("ML_DB.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Model1")
        rows = cursor.fetchall()
        conn.close()
        for row in rows:
            self.mgmt_tree.insert("", "end", values=row)
    def refresh_vendor_records(self):
        conn = sqlite3.connect("ML_DB.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Model2")
        rows = cursor.fetchall()
        conn.close()
        for row in rows:
            self.vendor_tree.insert("", "end", values=row)
    def delete_selected_record(self):
        selected_item = self.tree.selection()[0]
        values = self.tree.item(selected_item, "values")
        username_to_delete = values[0]
        
        conn = sqlite3.connect("ML_DB.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM Registration WHERE User_Name=?", (username_to_delete,))
        conn.commit()
        conn.close()
        
        self.tree.delete(selected_item)
        messagebox.showinfo("Success", "Record deleted successfully!")
    
    def upload_image(self):
        self.file_path=filedialog.askopenfilename()
        if self.file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0,self.file_path)

            image = Image.open(self.file_path)
            image = image.resize((150,200), Image.LANCZOS)  # Changed from Image.ANTIALIAS to Image.LANCZOS
            photo = ImageTk.PhotoImage(image)

            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Keep reference to avoid garbage collection
    def upload_image1(self):
        self.file_path=filedialog.askopenfilename()
        if self.file_path:
            self.file_entry_vend.delete(0, tk.END)
            self.file_entry_vend.insert(0,self.file_path)

            image = Image.open(self.file_path)
            image = image.resize((150,200), Image.LANCZOS)  # Changed from Image.ANTIALIAS to Image.LANCZOS
            photo = ImageTk.PhotoImage(image)

            self.image_label_vend.configure(image=photo)
            self.image_label_vend.image = photo  # Keep reference to avoid garbage collection
#######################################################################################################################
    def train_model(self):
        self.text_area.insert(tk.END, "..............................................................................\n")
        self.text_area.insert(tk.END, "Training model...\n")
        print(self.file_path)
        # Load MNIST dataset (example loading MNIST digits dataset)
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values
        
        # Create CNN model
        model = self.create_cnn_model()
        
        try:
            # Train the model
            self.training_callback = TrainingCallback(self.text_area)
            history = model.fit(
                x_train, y_train,
                epochs=10,
                validation_data=(x_test, y_test),
                callbacks=[self.training_callback]
            )
            
           
            
            # Display training metrics
            train_loss = history.history['loss'][-1]
            train_accuracy = history.history['accuracy'][-1]
            self.text_area.insert(tk.END, f"Final Train Loss: {train_loss:.4f}\n")
            self.text_area.insert(tk.END, f"Final Train Accuracy: {train_accuracy:.4f}\n")
            self.text_area.insert(tk.END, "..............................................................................\n")
            self.text_area.insert(tk.END, "\n")
            
            self.text_area.insert(tk.END, "\n=======================================================================================.\n")
            self.text_area.insert(tk.END, "Training completed.\n")
            # Plot training and validation metrics
            for epoch in range(10):
                epoch_info = f"Epoch {epoch+1}: "
                epoch_info += f"loss={history.history['loss'][epoch]:.4f}, "
                epoch_info += f"accuracy={history.history['accuracy'][epoch]:.4f}, "
                epoch_info += f"val_loss={history.history['val_loss'][epoch]:.4f}, "
                epoch_info += f"val_accuracy={history.history['val_accuracy'][epoch]:.4f}\n"
                
                self.text_area.insert(tk.END, epoch_info,"\n")
            self.text_area.insert(tk.END,"\n")
            
            # Now, integrate the matching of the uploaded image to MNIST dataset
            if self.file_path:
                uploaded_image=Image.open(self.file_path).convert('L')  # Convert to grayscale
                uploaded_image=uploaded_image.resize((28, 28))  # Resize to MNIST image size
                uploaded_image=np.array(uploaded_image)  # Convert to numpy array
                uploaded_image=uploaded_image / 255.0  # Normalize pixel values

                # Reshape for model prediction (add batch dimension)
                uploaded_image = uploaded_image.reshape(1, 28, 28, 1)

                # Predict the digit
                prediction = model.predict(uploaded_image)
                predicted_class = np.argmax(prediction)

                self.text_area.insert(tk.END, f"Predicted Digit: {predicted_class}\n")
                self.text_area.insert(tk.END, "..............................................................................\n")
                self.plot_metrics(history)

        except Exception as e:
            self.text_area.insert(tk.END, f"Training failed: {str(e)}\n")


    def create_cnn_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def plot_metrics(self, history):
        # Create a new figure with two subplots
        figure = Figure(figsize=(12, 6), dpi=100)
        
        # Add the subplots
        ax1 = figure.add_subplot(1, 2, 1)  # For Loss
        ax2 = figure.add_subplot(1, 2, 2)  # For Accuracy

        # Define the epochs
        epochs = range(1, len(history.history['accuracy']) + 1)

        # Plot the Loss metrics
        ax1.plot(epochs, history.history['loss'], label='Train Loss')
        ax1.plot(epochs, history.history['val_loss'], label='Val Loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot the Accuracy metrics
        ax2.plot(epochs, history.history['accuracy'], label='Train Accuracy')
        ax2.plot(epochs, history.history['val_accuracy'], label='Val Accuracy')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        # Create a FigureCanvasTkAgg object and add it to the Tkinter widget
        self.canvas = FigureCanvasTkAgg(figure, self.text_area)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        self.text_area.insert(tk.END, "\n")


    def test_model(self):
        self.text_area.insert(tk.END, "Testing model with hyperparameter tuning...\n")
        
        self.text_area.insert(tk.END, "..............................................................................\n")
        self.text_area.insert(tk.END, "Training model...\n")
        
        # Load MNIST dataset (example loading MNIST digits dataset)
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values
        
        # Create CNN model
        model = self.create_cnn_model()
        
        try:
            self.training_callback = TrainingCallback(self.text_area)
            # Train the model
            history = model.fit(
                x_train, y_train,
                epochs=10,
                validation_data=(x_test, y_test),
                callbacks=[self.training_callback]
            )
            
            self.text_area.insert(tk.END, "Training completed.\n")
            
            # Display training metrics
            train_loss = history.history['loss'][-1]
            train_accuracy = history.history['accuracy'][-1]
            self.text_area.insert(tk.END, f"Final Train Loss: {train_loss:.4f}\n")
            self.text_area.insert(tk.END, f"Final Train Accuracy: {train_accuracy:.4f}\n")
            self.text_area.insert(tk.END, "..............................................................................\n")
            self.text_area.insert(tk.END, "\n")

            # Plot training and validation metrics
            self.text_area.insert(tk.END, "\n=======================================================================================.\n")
            self.text_area.insert(tk.END, "Training completed.\n")
            # Plot training and validation metrics
            for epoch in range(10):
                epoch_info = f"Epoch {epoch+1}: "
                epoch_info += f"loss={history.history['loss'][epoch]:.4f}, "
                epoch_info += f"accuracy={history.history['accuracy'][epoch]:.4f}, "
                epoch_info += f"val_loss={history.history['val_loss'][epoch]:.4f}, "
                epoch_info += f"val_accuracy={history.history['val_accuracy'][epoch]:.4f}\n"
                
                self.text_area.insert(tk.END, epoch_info,"\n")
            self.text_area.insert(tk.END,"\n")
            #////////////////
            # Now, integrate the matching of the uploaded image to MNIST dataset
            if self.file_path:
                uploaded_image = Image.open(self.file_path).convert('L')  # Convert to grayscale
                uploaded_image = uploaded_image.resize((28, 28))  # Resize to MNIST image size
                uploaded_image = np.array(uploaded_image)  # Convert to numpy array
                uploaded_image = uploaded_image / 255.0  # Normalize pixel values

                # Reshape for model prediction (add batch dimension)
                uploaded_image = uploaded_image.reshape(1, 28, 28, 1)

                # Predict the digit
                prediction = model.predict(uploaded_image)
                predicted_class = np.argmax(prediction)

                self.text_area.insert(tk.END, f"Predicted Digit: {predicted_class}\n")
                
                # Determine status of the image (real or fake) based on the predicted digit
                if predicted_class in [0, 1, 2, 3, 4]:  # Example: If predicted digit is 0-4, consider it real
                    status = "Real Image"
                else:
                    status = "Fake Image"
                
                self.text_area.insert(tk.END, f"Status: {status}\n")
                self.text_area.insert(tk.END, "..............................................................................\n")
                
                # Store status in fake_real_field1 or any appropriate variable
                self.fake_real_field.insert(tk.END,status)  # Assuming fake_real_field1 is a class attribute
                self.plot_metrics(history)

        except Exception as e:
            self.text_area.insert(tk.END, f"Training failed: {str(e)}\n")
        

    
####################################################################################
   
####################################################################################################################################
    def save_record(self):
        self.s1 = self.file_entry.get()
        self.s2 = self.fake_real_field.get()  # Assuming fake_real_field is a method or attribute that returns a value
        
        conn = sqlite3.connect("ML_DB.db")
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Model1 (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Image_path TEXT,
                Image_Status TEXT
            )
        """)
        
        # Insert data into the table
        cursor.execute("INSERT INTO Model1(Image_path, Image_Status) VALUES (?,?)", (self.s1, self.s2))
        
        conn.commit()
        conn.close()
        
        messagebox.showinfo("Success", "Model Details Added Successfully to DB!")

    def logout1(self):
        self.show_frame("LoginFrame")
############################################################################################################
#####################(for the vendor)###############################
    def save_record1(self):
        self.s1 = self.file_entry_vend.get()
        self.s2 = self.fake_real_field1.get()  # Assuming fake_real_field is a method or attribute that returns a value
        
        conn = sqlite3.connect("ML_DB.db")
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Model2 (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Image_path TEXT,
                Image_Status TEXT
            )
        """)
        
        # Insert data into the table
        cursor.execute("INSERT INTO Model2(Image_path, Image_Status) VALUES (?, ?)", (self.s1, self.s2))
        
        conn.commit()
        conn.close()
        
        messagebox.showinfo("Success", "Model Details Added Successfully to DB!")
    def logout_btn(self):
        self.show_frame("LoginFrame")
    def train_model1(self):
        self.text_area_vend.insert(tk.END, "Training model...\n")
        self.text_area.insert(tk.END, "..............................................................................\n")
        self.text_area.insert(tk.END, "Training model...\n")
        print(self.file_path)
        # Load MNIST dataset (example loading MNIST digits dataset)
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values
        
        # Create CNN model
        model = self.create_cnn_model()
        
        try:
            self.training_callback = TrainingCallback(self.text_area)
            # Train the model
            history = model.fit(
                x_train, y_train,
                epochs=10,
                validation_data=(x_test, y_test),
                callbacks=[self.training_callback]
            )
            
            self.text_area.insert(tk.END, "Training completed.\n")
            
            # Display training metrics
            train_loss = history.history['loss'][-1]
            train_accuracy = history.history['accuracy'][-1]
            self.text_area.insert(tk.END, f"Final Train Loss: {train_loss:.4f}\n")
            self.text_area.insert(tk.END, f"Final Train Accuracy: {train_accuracy:.4f}\n")
            self.text_area.insert(tk.END, "..............................................................................\n")
            self.text_area.insert(tk.END, "\n")

            # Plot training and validation metrics
            ##################################################################################
            self.text_area.insert(tk.END, "\n=======================================================================================.\n")
            self.text_area.insert(tk.END, "Training completed.\n")
            # Plot training and validation metrics
            for epoch in range(10):
                epoch_info = f"Epoch {epoch+1}: "
                epoch_info += f"loss={history.history['loss'][epoch]:.4f}, "
                epoch_info += f"accuracy={history.history['accuracy'][epoch]:.4f}, "
                epoch_info += f"val_loss={history.history['val_loss'][epoch]:.4f}, "
                epoch_info += f"val_accuracy={history.history['val_accuracy'][epoch]:.4f}\n"
                
                self.text_area.insert(tk.END, epoch_info,"\n")
            self.text_area.insert(tk.END,"\n")
            # Now, integrate the matching of the uploaded image to MNIST dataset
            if self.file_path:
                uploaded_image=Image.open(self.file_path).convert('L')  # Convert to grayscale
                uploaded_image=uploaded_image.resize((28, 28))  # Resize to MNIST image size
                uploaded_image=np.array(uploaded_image)  # Convert to numpy array
                uploaded_image=uploaded_image / 255.0  # Normalize pixel values

                # Reshape for model prediction (add batch dimension)
                uploaded_image = uploaded_image.reshape(1, 28, 28, 1)

                # Predict the digit
                prediction = model.predict(uploaded_image)
                predicted_class = np.argmax(prediction)

                self.text_area.insert(tk.END, f"Predicted Digit: {predicted_class}\n")
                self.text_area.insert(tk.END, "..............................................................................\n")
                self.plot_metrics(history)

        except Exception as e:
            self.text_area.insert(tk.END, f"Training failed: {str(e)}\n")
    def test_model1(self):
        self.text_area_vend.insert(tk.END, "Testing model...\n")
        self.text_area.insert(tk.END, "Testing model with hyperparameter tuning...\n")
        
        self.text_area.insert(tk.END, "..............................................................................\n")
        self.text_area.insert(tk.END, "Training model...\n")
        
        # Load MNIST dataset (example loading MNIST digits dataset)
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values
        
        # Create CNN model
        model = self.create_cnn_model()
        
        try:
            self.training_callback = TrainingCallback(self.text_area)
            # Train the model
            history = model.fit(
                x_train, y_train,
                epochs=10,
                validation_data=(x_test, y_test),
                callbacks=[self.training_callback]
            )
            
            self.text_area.insert(tk.END, "Training completed.\n")
            
            # Display training metrics
            train_loss = history.history['loss'][-1]
            train_accuracy = history.history['accuracy'][-1]
            self.text_area.insert(tk.END, f"Final Train Loss: {train_loss:.4f}\n")
            self.text_area.insert(tk.END, f"Final Train Accuracy: {train_accuracy:.4f}\n")
            self.text_area.insert(tk.END, "..............................................................................\n")
            self.text_area.insert(tk.END, "\n")

            # Plot training and validation metrics
            ##################################################################################
            self.text_area.insert(tk.END, "\n=======================================================================================.\n")
            self.text_area.insert(tk.END, "Training completed.\n")
            # Plot training and validation metrics
            for epoch in range(10):
                epoch_info = f"Epoch {epoch+1}: "
                epoch_info += f"loss={history.history['loss'][epoch]:.4f}, "
                epoch_info += f"accuracy={history.history['accuracy'][epoch]:.4f}, "
                epoch_info += f"val_loss={history.history['val_loss'][epoch]:.4f}, "
                epoch_info += f"val_accuracy={history.history['val_accuracy'][epoch]:.4f}\n"
                
                self.text_area.insert(tk.END, epoch_info,"\n")
            self.text_area.insert(tk.END,"\n")
            
            # Now, integrate the matching of the uploaded image to MNIST dataset
            if self.file_path:
                uploaded_image = Image.open(self.file_path).convert('L')  # Convert to grayscale
                uploaded_image = uploaded_image.resize((28, 28))  # Resize to MNIST image size
                uploaded_image = np.array(uploaded_image)  # Convert to numpy array
                uploaded_image = uploaded_image / 255.0  # Normalize pixel values

                # Reshape for model prediction (add batch dimension)
                uploaded_image = uploaded_image.reshape(1, 28, 28, 1)

                # Predict the digit
                prediction = model.predict(uploaded_image)
                predicted_class = np.argmax(prediction)

                self.text_area.insert(tk.END, f"Predicted Digit: {predicted_class}\n")
                
                # Determine status of the image (real or fake) based on the predicted digit
                if predicted_class in [0, 1, 2, 3, 4]:  # Example: If predicted digit is 0-4, consider it real
                    status = "Real Image"
                else:
                    status = "Fake Image"
                
                self.text_area.insert(tk.END, f"Status: {status}\n")
                self.text_area.insert(tk.END, "..............................................................................\n")
                
                # Store status in fake_real_field1 or any appropriate variable
                self.fake_real_field.insert(tk.END,status)  # Assuming fake_real_field1 is a class attribute
                self.plot_metrics(history)
        except Exception as e:
            self.text_area.insert(tk.END, f"Training failed: {str(e)}\n")
#################################################################################################
class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, text_area):
        super().__init__()
        self.text_area = text_area

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_info = f"Epoch {epoch+1}: "
        epoch_info += f"loss={logs.get('loss', 0):.4f}, "
        epoch_info += f"accuracy={logs.get('accuracy', 0):.4f}, "
        epoch_info += f"val_loss={logs.get('val_loss', 0):.4f}, "
        epoch_info += f"val_accuracy={logs.get('val_accuracy', 0):.4f}\n"
        
        self.text_area.insert(tk.END, epoch_info)
        self.text_area.see(tk.END)  # Scroll to the end
############################################################################################################################################
if __name__ == "__main__":
    app = AnimatedLoginForm()
    app.mainloop()
