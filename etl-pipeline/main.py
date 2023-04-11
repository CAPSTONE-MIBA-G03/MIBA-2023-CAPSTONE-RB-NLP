from link_extractor import Google

google = Google(company="UBS")
google_links = google.get_links(max_articles=1)