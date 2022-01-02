output = lin-text-classifier

submit: clean
	bundle install; bundle exec jekyll build
	mkdir -p $(output)
	cp -r _site/* $(output)
	sed -i 's/"\/text-classifier\//"/g' $(output)/index.html
	cp -r scaffold/ $(output)
	zip $(output).zip -r -9 $(output)

clean:
	rm -rf $(output) $(output).zip
