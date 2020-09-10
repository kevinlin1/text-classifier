output = lin-text-classifier

nifty: clean
	cd docs/; bundle install; bundle exec jekyll build
	mkdir -p $(output)
	cp -r docs/_site/* $(output)
	sed -i 's/"\/text-classifier\//"/g' $(output)/index.html
	cp -r scaffold/ $(output)
	zip $(output).zip -r -9 $(output)

clean:
	rm -rf $(output) $(output).zip
