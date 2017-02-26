#!/usr/bin/env python2.7
"""
This script checks whether this instance will be terminated. Then,
it sends a slack message.
"""
import argparse
import time
import subprocess


CURL = '/usr/bin/curl'
SLACK_URL_REVERSE = 'GRMf5GykAJfVFCrv3guujtre/VJ4FG034B/ETNN38Z3T/secivres/moc.kcals.skooh//:sptth'
TERMINATION_LINK = 'http://169.254.169.254/latest/meta-data/spot/termination-time'
NOT_FOUND = '404 - Not Found'


def get_parser():
    parser = argparse.ArgumentParser(description='ec2 terminate checker.')
    parser.add_argument('-c', '--channel',
                        nargs='?',
                        default='#random',
                        help='slack channel')
    parser.add_argument('-t', '--text',
                        nargs='?',
                        default='Instance is getting terminated soon. Please request spot instance again.',
                        help='slack text')
    parser.add_argument('-u', '--username',
                        nargs='?',
                        default='sangmin\'s model 2 with tensorflow',
                        help='slack username')
    parser.add_argument('-f', '--frequency',
                        nargs='?',
                        default=5, type=int,
                        help='check frequency in second')
    return parser


def is_terminated():
    """
    Check termination of this instance

    .. Amazon doc: http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-interruptions.html
    """
    args = [CURL, '-s', TERMINATION_LINK]
    output = subprocess.check_output(args)
    return output.index(NOT_FOUND) <= 0


def send_slack_message(channel, username, text):
    """
    Send slack message

    :param channel: slack channel
    :param username: slack username
    :param text: slack text
    """
    payload = 'payload={{"channel": "{0}", "username": "{1}", "text": "{2}"}}'.format(channel, username, text)
    slack_url = SLACK_URL_REVERSE[::-1]
    args = [CURL, '-X', 'POST', '--data-urlencode', payload, slack_url]
    subprocess.check_call(args)


def check_loop(frequency, channel, username, text):
    """
    Check termination with a loop

    :param frequency: frequency of loop in second
    :param channel: slack channel
    :param username: slack username
    :param text: slack text
    """
    while True:
        if is_terminated():
            send_slack_message(channel, username, text)
            return
        time.sleep(frequency)


def main():
    parser = get_parser()
    args = parser.parse_args()
    check_loop(args.frequency, args.channel, args.username, args.text)


if __name__ == '__main__':
    main()
