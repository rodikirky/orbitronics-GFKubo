import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.captureWarnings(True) # optional: route warnings -> logging