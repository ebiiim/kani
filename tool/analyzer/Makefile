all:
	pip install -r requirements.txt
	pyinstaller --onefile --hidden-import=matplotlib --hidden-import=tkinter --name=analyzer main.py
	cp -r example dist
