def clean_subs(subs):
    """
    Takes messy subs dicts and cleans it up.
    :param subs: Substitutions dictionary
    :return: Dict with proper uniting
    """
    cleaned_subs = {}
    vks = subs.keys()
    for vk in vks:
        try:
            cleaned_subs[vk] = subs[vk].value
        except:
            cleaned_subs[vk] = subs[vk]
    return cleaned_subs
