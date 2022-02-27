output = materials

submit: clean
	bundle install; bundle exec jekyll build
	mkdir -p $(output)
	cp -r _site/* $(output)
	sed -i 's/"\/text-classifier\//"/g' $(output)/index.html
	pandoc README.md -o $(output)/instructor-guide.docx
	cp -r scaffold/ $(output)
	zip $(output).zip -r -9 $(output)

clean:
	rm -rf $(output) $(output).zip
